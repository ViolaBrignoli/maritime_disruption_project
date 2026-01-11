"""
MARITIME DISRUPTION PROJECT - Cascade Simulation & Stress-Test Orchestrator

Provide the orchestration logic and helper utilities to run cascade-style
stress tests for port disruption scenarios. This module prepares panel
features for prediction, reconciles model input shapes, applies calibrated
shock injections (direct + neighbor propagation), evaluates the impact
using a persisted primary predictor, and augments results with an optional
XGBoost-based confidence model. It is intended to be called by a top-level
execution script (e.g., main.py) to run batches of scenarios and return a
structured list of affected ports and related metadata.

Key behaviours:
    - Build prediction input matrices from panel data:
        * Aggregates dynamic features by join_key (mean over prediction window)
        * Preserves/recovers static meta fields (lat/lon/country) for downstream
          reporting and spatial operations
        * Produces a canonical feature list used for alignment with the
          preprocessor and model
    - Reconcile encoded feature matrices to the model's expected input shape:
        * Detects a single unexpected encoded column and safely drops it
        * Drops zero-variance encoded columns if needed
        * Pads or truncates encoded matrices as a last resort with clear logging
    - Shock injection & neighbor propagation:
        * Calibrate injection magnitude for well-known hubs (configurable)
        * Inject shock into event features (events__n_events) of a chosen port
        * Propagate decayed shocks to nearest neighbors using a distance-based
          decay kernel (KNN-like spatial propagation)
        * Compute post-injection risk predictions using the primary model
    - Cascade result preparation:
        * Compute base and post-injection probabilities, risk deltas, and
          return an internal list of all ports with any non-trivial increase
          in predicted risk (minimal delta threshold applied)
    - Confidence metadata augmentation:
        * Optionally loads a persisted XGBoost "confidence" model to compute
          an independent probability and assign human-readable confidence tiers
        * The confidence model is used for metadata only (no filtering) so
          downstream logic can decide how to use confidence information
    - Batch orchestration:
        * Loads the persisted primary model artifact and runs multiple
          scenario simulations (shock sources) in a single batch
        * Returns a sorted list of enriched results (descending Risk_Increase)
        * Keeps IO / display responsibilities to the caller (main.py)
"""
from pathlib import Path
from typing import Any, List, Tuple, Dict
import traceback
import joblib
import numpy as np
import pandas as pd
import json

# Configuration
RESULTS_DIR = Path("results")
PREDICTOR_MODEL_PATH = RESULTS_DIR / "best_model_final.joblib"
XGB_CONFIDENCE_MODEL_PATH = RESULTS_DIR / "xgb_model_final.joblib" 

# Simulation parameters
INJECTION_VALUE = 5.0          # Default shock magnitude
NEIGHBOR_K = 50                # Number of nearest neighbors
NEIGHBOR_DECAY_FACTOR = 0.6    # Relative maximum magnitude for neighbors
NEIGHBOR_DISTANCE_SCALE = 1.0  # Scaling factor for distance-based decay

# Confidence and Economic Thresholds (Kept for Metadata)
CONFIDENCE_THRESHOLDS = {
    "High_Confidence": 0.15,      
    "Moderate_Confidence": 0.08,  
    "Low_Confidence": 0.03        
}
MIN_RISK_INCREASE = 0.02  # 2% increase threshold 

# Helper / Template Builder 
def get_base_features_template(df_panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """ Build the raw (unencoded) feature matrix aggregated over the prediction window. """
    DYNAMIC_PREFIXES = ("activity__", "chokepoints__", "plsci__", "piracy__", "events__", "network__", "country__")
    STATIC_EXPLICIT = ["meta__lat", "meta__lon", "meta__vessel_count_total", "meta__vessel_count_container"]

    dynamic_cols = [c for c in df_panel.columns if c.startswith(DYNAMIC_PREFIXES) and "disrupted_flag" not in c and "target_" not in c and c not in ("join_key", "year")]
    categorical_candidates = [c for c in df_panel.columns if (c.startswith(("meta__country", "meta__industry", "meta__iso3")) or df_panel[c].dtype == "object") and c not in ("join_key", "year")]

    df_future = df_panel[df_panel["year"] >= 2021].copy().reset_index(drop=True)

    dyn_present = [c for c in dynamic_cols if c in df_future.columns]
    if dyn_present:
        df_dyn = df_future.groupby("join_key")[dyn_present].mean().reset_index()
    else:
        df_dyn = df_future[["join_key"]].drop_duplicates().reset_index(drop=True)

    static_cols = [c for c in df_future.columns if c in STATIC_EXPLICIT + categorical_candidates]
    agg_map = {}
    for c in static_cols:
        if c in STATIC_EXPLICIT:
            agg_map[c] = "first"
        else:
            agg_map[c] = lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown"

    if agg_map:
        df_static = df_future.groupby("join_key").agg(agg_map).reset_index()
    else:
        df_static = df_future[["join_key"]].drop_duplicates().reset_index(drop=True)

    df_merge = df_dyn.merge(df_static, on="join_key", how="left")

    for meta_col in ("meta__country", "meta__lat", "meta__lon"):
        if meta_col not in df_merge.columns:
            if meta_col in df_panel.columns:
                fallback = df_panel.groupby("join_key")[meta_col].first().reset_index()
                df_merge = df_merge.merge(fallback, on="join_key", how="left")
            else:
                df_merge[meta_col] = "Unknown" if meta_col == "meta__country" else 0.0

    relevant = [c for c in (dyn_present + [c for c in STATIC_EXPLICIT if c in df_merge.columns] + categorical_candidates) if c in df_merge.columns]
    for forbidden in ("join_key", "year", "port_name", "index"):
        if forbidden in relevant:
            relevant = [c for c in relevant if c != forbidden]

    cols = ["join_key"] + relevant
    df_merge = df_merge[[c for c in cols if c in df_merge.columns]]

    df_meta = df_merge[["join_key", "meta__country", "meta__lat", "meta__lon"]].copy()
    df_meta["join_key"] = df_meta["join_key"].astype(str)
    df_meta = df_meta.reset_index(drop=True)

    return df_merge.drop(columns=["join_key"], errors="ignore"), df_meta, relevant
# Helper functions for reconciliation
def _drop_single_extra_encoded_column_by_name(X_enc: np.ndarray, enc_names: List[str], expected_encoded_set: set, verbose: bool = True):
    extras = [n for n in enc_names if n not in expected_encoded_set]
    if len(extras) == 1:
        idx = enc_names.index(extras[0])
        mask = [i for i in range(X_enc.shape[1]) if i != idx]
        X_new = X_enc[:, mask]
        new_names = [enc_names[i] for i in mask]
        if verbose:
            print(f"   (i) Dropping single unexpected encoded feature '{extras[0]}' (safe).")
        return X_new, new_names
    return None, None

def _drop_zero_variance_columns(X_enc: np.ndarray, enc_names: List[str], n_keep: int):
    col_var = np.var(X_enc, axis=0)
    zero_var_idx = [i for i, v in enumerate(col_var) if np.isclose(v, 0.0)]
    if not zero_var_idx:
        return None, None
    n_current = X_enc.shape[1]
    need_drop = n_current - n_keep
    to_drop = zero_var_idx[:need_drop] if need_drop <= len(zero_var_idx) else zero_var_idx
    mask = [i for i in range(n_current) if i not in to_drop]
    X_new = X_enc[:, mask]
    new_names = [enc_names[i] for i in mask]
    return X_new, new_names

def _reconcile_encoded_matrix(
    X_encoded: np.ndarray,
    encoded_names: List[str],
    model_expected_n: int,
    feature_names_cleaned: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """ Reconcile X_encoded with model_expected_n. """
    n_enc = X_encoded.shape[1]
    enc_names = list(encoded_names) if encoded_names is not None else [f"enc_{i}" for i in range(n_enc)]
    target = model_expected_n

    if n_enc == target:
        return X_encoded, enc_names

    if feature_names_cleaned:
        expected_set = set(feature_names_cleaned)
        dropped_result = _drop_single_extra_encoded_column_by_name(X_encoded, enc_names, expected_set, verbose=True)
        if dropped_result[0] is not None:
            X_new, names_new = dropped_result
            if X_new.shape[1] == target:
                return X_new, names_new
            X_encoded, enc_names = X_new, names_new
            n_enc = X_encoded.shape[1]

    if n_enc > target:
        drop_zero = _drop_zero_variance_columns(X_encoded, enc_names, target)
        if drop_zero[0] is not None:
            X_encoded, enc_names = drop_zero
            if X_encoded.shape[1] == target:
                print(f"   (i) Dropped zero-variance encoded columns to match expected shape ({target}).")
                return X_encoded, enc_names

    if X_encoded.shape[1] > target:
        n_drop = X_encoded.shape[1] - target
        print(f"   (warn) Will drop {n_drop} encoded columns (last-resort).")
        X_encoded = X_encoded[:, :target]
        enc_names = enc_names[:target]
        return X_encoded, enc_names

    if X_encoded.shape[1] < target:
        n_pad = target - X_encoded.shape[1]
        print(f"   (warn) Will pad encoded matrix with {n_pad} zero columns.")
        X_encoded = np.hstack([X_encoded, np.zeros((X_encoded.shape[0], n_pad))])
        enc_names = enc_names + [f"pad_{i}" for i in range(n_pad)]
        return X_encoded, enc_names

    return X_encoded, enc_names

# Shock Magnitude Calibration
def calibrate_shock_magnitude(port_name: str, base_shock: float = 5.0) -> float:
    """
    Calibrate shock magnitude based on port characteristics.
    """
    major_hubs = {
        'shanghai': 8.0,
        'singapore': 6.0,
        'rotterdam': 5.5,
        'suez': 7.0,
        'durban': 6.5,
        'ningbo': 7.5,
        'hong kong': 6.5,
        'busan': 6.0,
        'los angeles': 6.5,
        'long beach': 6.0,
        'antwerp': 5.5,
        'hamburg': 5.5,
        'port said': 6.5,
        'jebel ali': 6.0,
        'kaohsiung': 5.5
    }
    
    port_lower = port_name.lower()
    for hub, shock in major_hubs.items():
        if hub in port_lower:
            return shock
    
    return base_shock  

# Cascade simulation (with neighbor propagation) 
def run_cascading_prediction(
    model: Any,
    X_encoded: np.ndarray,
    df_meta: pd.DataFrame,
    disrupted_port_name: str,
    encoded_feature_names: List[str],
    shock_magnitude: float = INJECTION_VALUE,
) -> List[Dict]:
    """
    Runs a cascading simulation using Primary Model (KNN) and returns ALL affected ports.
    """
    if model is None:
        raise RuntimeError("run_cascading_prediction requires a trained model instance.")

    match = df_meta[df_meta["join_key"].str.contains(disrupted_port_name, case=False)]
    if match.empty:
        print(f"   (x) ERROR: Could not locate disrupted port '{disrupted_port_name}' in metadata.")
        return []

    disrupted_join_key = match["join_key"].iloc[0]
    disrupted_idx = int(df_meta.index[df_meta["join_key"] == disrupted_join_key][0])

    event_indices = [i for i, n in enumerate(encoded_feature_names or []) if "events__n_events" in n]

    try:
        base_probs = model.predict_proba(X_encoded)[:, 1]
    except Exception as e:
        print(f"   (x) ERROR: model.predict_proba on base matrix failed: {e}")
        return []

    X_sim = X_encoded.copy()

    if event_indices:
        for ei in event_indices:
            X_sim[disrupted_idx, ei] = shock_magnitude 

        try:
            vec = X_encoded[disrupted_idx].astype(float)
            diffs = X_encoded.astype(float) - vec
            dists = np.linalg.norm(diffs, axis=1)
            dists[disrupted_idx] = np.inf 
            n_ports = X_encoded.shape[0]
            k = min(NEIGHBOR_K, n_ports - 1)
            neighbors = np.argsort(dists)[:k]
            neighbor_dists = dists[neighbors]
            stdd = float(np.std(neighbor_dists))
            if stdd <= 0 or np.isnan(stdd):
                stdd = float(np.mean(neighbor_dists)) if np.mean(neighbor_dists) > 0 else 1.0
            decay = np.exp(-(neighbor_dists / (stdd * NEIGHBOR_DISTANCE_SCALE + 1e-12)))
            if decay.max() > 0:
                decay = decay / decay.max()
            neighbor_inj = shock_magnitude * NEIGHBOR_DECAY_FACTOR * decay
            for ni, inj in zip(neighbors, neighbor_inj):
                for ei in event_indices:
                    X_sim[ni, ei] += inj
        except Exception as e:
            print(f"   (warn) Neighbor propagation failed: {e}")
    else:
        print("   (warn) No encoded event columns found; injection skipped.")

    try:
        post_probs = model.predict_proba(X_sim)[:, 1]
    except Exception as e:
        print(f"   (x) ERROR: model.predict_proba on simulated matrix failed: {e}")
        return []

    df_res = df_meta.copy()
    df_res["Base_Risk_Prob"] = base_probs
    df_res["Cascade_Risk_Prob"] = post_probs
    df_res["Risk_Delta"] = df_res["Cascade_Risk_Prob"] - df_res["Base_Risk_Prob"]
    df_res["Scenario_Source"] = disrupted_port_name
    df_res = df_res[df_res["join_key"] != disrupted_join_key] 
    
    # Keeping only ports with *any* risk increase (0.0001) 
    MIN_CASCADE_DELTA = 0.0001
    out = []
    for index, r in df_res.iterrows():
        risk_delta = r["Risk_Delta"]
        
        if risk_delta >= MIN_CASCADE_DELTA:
            out.append({
                "Port": r["join_key"],
                "Country": r["meta__country"],
                "Original_Risk": r["Base_Risk_Prob"],
                "Cascade_Risk": r["Cascade_Risk_Prob"],
                "Risk_Increase": r["Risk_Delta"],
                "Scenario_Source": r["Scenario_Source"],
                "Index": index 
            })
    
    print(f"   (i) Cascade affected {len(out)} ports (filtered for risk_increase >= {MIN_CASCADE_DELTA:.4f})")
    return out


def xgb_confidence_filter(
    X_encoded_full: np.ndarray, 
    cascade_results: List[Dict], 
    encoded_feature_names: List[str]
) -> List[Dict]:
    """
    Screens cascade results using XGBoost for confidence Metadata
    """
    
    # Initialize default confidence fields
    for res in cascade_results:
        res["XGB_Risk_Prob"] = 0.0 # Default value
        res["Confidence_Tier"] = "Confidence_N/A"
        res["Is_High_Confidence"] = False
        res["Is_Actionable"] = False

    if not XGB_CONFIDENCE_MODEL_PATH.exists():
        print(f"   (x) ERROR: XGBoost confidence model not found. Skipping detailed confidence analysis.")
        print(f"   (v) Returning {len(cascade_results)} unfiltered predictions (without XGB data).")
        return cascade_results

    try:
        xgb_model = joblib.load(XGB_CONFIDENCE_MODEL_PATH)
        print(f"   (i) Loaded secondary confidence model (XGBoost).")
    except Exception as e:
        print(f"   (x) ERROR loading XGBoost model: {e}")
        print(f"   (v) Returning {len(cascade_results)} unfiltered predictions (without XGB data).")
        return cascade_results

    # Get indices and prepare input
    indices = [r['Index'] for r in cascade_results]
    X_xgb = X_encoded_full[indices, :]
    
    # Predict with XGBoost
    try:
        xgb_probs = xgb_model.predict_proba(X_xgb)[:, 1]
    except Exception as e:
        print(f"   (x) ERROR predicting XGBoost confidence: {e}")
        print(f"   (v) Returning {len(cascade_results)} unfiltered predictions (without XGB data).")
        return cascade_results

    # Apply metadata generation
    stats = {"high": 0, "moderate": 0, "low": 0}
    
    for i, res in enumerate(cascade_results):
        xgb_prob = xgb_probs[i]
        risk_increase = float(res["Risk_Increase"])
        
        # Determine confidence tier (Metadata only)
        if xgb_prob >= CONFIDENCE_THRESHOLDS["High_Confidence"]:
            confidence_tier = "High_Confidence"
            stats["high"] += 1
        elif xgb_prob >= CONFIDENCE_THRESHOLDS["Moderate_Confidence"]:
            confidence_tier = "Moderate_Confidence"
            stats["moderate"] += 1
        elif xgb_prob >= CONFIDENCE_THRESHOLDS.get("Low_Confidence", 0.03):
            confidence_tier = "Low_Confidence"
            stats["low"] += 1
        else:
            confidence_tier = "Very_Low_Confidence"

        # Store enhanced metadata 
        res["XGB_Risk_Prob"] = xgb_prob
        res["Confidence_Tier"] = confidence_tier
        res["Is_High_Confidence"] = (confidence_tier == "High_Confidence")
        res["Is_Actionable"] = (risk_increase >= MIN_RISK_INCREASE)
    
    print(f"   (i) Confidence distribution: High={stats['high']}, Moderate={stats['moderate']}, Low={stats['low']}")
    print(f"   (v) Returning {len(cascade_results)} total predictions (FILTRATION REMOVED).")
    
    return cascade_results

# Main orchestration entry for Batch Execution
def execute_batch_stress_test(
    df_panel: pd.DataFrame,
    feature_names_cleaned: List[str],
    preprocessor: Any,
    scenarios: List[Dict[str, Any]],
) -> List[Dict]:
    """
    Orchestrates, running and consolidating results from multiple scenarios,
    and returns ALL affected ports sorted by Risk_Increase.
    """
    print("\n" + "-" * 100)
    print("BATCH STRESS TEST PREPARATION")
    print("-" * 100)

    # Load primary model
    if not PREDICTOR_MODEL_PATH.exists():
        print("   (x) ERROR: Trained primary model artifact not found.")
        return []
    try:
        primary_model = joblib.load(PREDICTOR_MODEL_PATH)
    except Exception as e:
        print(f"   (x) ERROR loading primary model: {e}")
        traceback.print_exc()
        return []
        
    model_expected_n = getattr(primary_model, "n_features_in_", None)
    
    # Data Prep (Template, Align, Encode, Reconcile)
    try:
        df_raw, df_meta, relevant_features = get_base_features_template(df_panel)
        expected_input_features = getattr(preprocessor, "input_feature_names_", list(relevant_features))
        
        for feat in expected_input_features:
            if feat not in df_raw.columns:
                df_raw[feat] = "Unknown" if feat.startswith(("meta__country")) else 0.0

        df_align = df_raw[expected_input_features].copy()
        X_encoded = preprocessor.transform(df_align)
        encoded_names = list(preprocessor.get_feature_names_out())
        
        final_expected = model_expected_n if model_expected_n is not None else (len(feature_names_cleaned) if feature_names_cleaned is not None else X_encoded.shape[1])
        
        if X_encoded.shape[1] != final_expected:
            print(f"   (warn) ENCODING produced {X_encoded.shape[1]} features but model expects {final_expected}. Reconciling...")
            X_encoded, encoded_names = _reconcile_encoded_matrix(X_encoded, encoded_names, final_expected, feature_names_cleaned)
        
    except Exception as e:
        print(f"   (x) ERROR during batch data prep/reconciliation: {e}")
        traceback.print_exc()
        return []

    # Run all simulations
    all_cascade_results = []
    
    print("\n" + "-" * 100)
    print(f"STARTING {len(scenarios)} SCENARIO RUNS (Propagation by KNN)")
    print("-" * 100)
    
    for scenario in scenarios:
        sim_port = scenario['port']
        shock = scenario.get('shock', None)
        
    
        if shock is None:
            shock = calibrate_shock_magnitude(sim_port)
        
        print(f"   -> Running Scenario: {sim_port} (Shock: {shock:.1f})")
        
        scenario_results = run_cascading_prediction(
            model=primary_model,
            X_encoded=X_encoded,
            df_meta=df_meta,
            disrupted_port_name=sim_port,
            encoded_feature_names=encoded_names,
            shock_magnitude=shock,
        )
        all_cascade_results.extend(scenario_results)

    print(f"Completed {len(scenarios)} cascades. Total results: {len(all_cascade_results)}.")
    
    # Run secondary confidence filter (XGBoost) on all aggregated results
    print("\n" + "-" * 100)
    print("CONFIDENCE SCREENING")
    print("-" * 100)
    
    final_results = xgb_confidence_filter(X_encoded, all_cascade_results, encoded_names)
    
    print(f"   (v) Confidence metadata applied to {len(all_cascade_results)} total predictions.")
    
    # Post-Processing: Sort by Risk_Increase (descending)
    final_results.sort(key=lambda x: x['Risk_Increase'], reverse=True)
    
    # Reformat numeric fields to strings before returning 
    for res in final_results:
        res['Risk_Increase'] = f"{res['Risk_Increase']:.4f}"
        # Ensure XGB_Risk_Prob is formatted
        res['XGB_Risk_Prob'] = f"{res['XGB_Risk_Prob']:.4f}" if isinstance(res['XGB_Risk_Prob'], (float, np.float64)) else res['XGB_Risk_Prob']

    # Return 
    return final_results
