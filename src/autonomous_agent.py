"""
MARITIME DISRUPTION PROJECT - Autonomous Resilience Agent (ARA): Monte Carlo Resource Allocation for Cascading Risk Reduction

This module implements a lightweight, self-contained Analytic Resource Allocation (ARA)
agent that recommends sequential investments to reduce systemic cascading risk. 
It uses a trained predictor model and a preprocessor to run
many forward simulations of cascading effects under stochastic shock scenarios, and
searches for the best incremental investments to lower expected total risk.

Primary Function:
run_ara_agent(df_panel, feature_names_cleaned, preprocessor, ...)
    Produces a list of investment steps each containing:
    - Step: sequential step number
    - Invest_Hub: chosen hub identifier (join_key / Port / name)
    - Risk_Reduction_Marginal: expected marginal reduction in systemic risk
    - Cumulative_Risk_Remaining: estimated mean residual system risk after commit
    - Cumulative_Reduction_Fraction: fraction of original event magnitude reduced
    - Policy_Confidence: heuristic confidence score (expected / std)

Configurable Parameters:
- INVESTMENT_BUDGET (int): number of investment steps to try (default 5).
- DEFAULT_MC_SAMPLES (int): Monte Carlo trials per evaluation (small default for interactivity).
- DEFAULT_SHOCK_NOISE_FRAC (float): relative noise sigma for shock perturbations.
- DEFAULT_N_JOBS (int): parallel jobs for coarse/refined evaluations (bounded by CPU).
- DEFAULT_TOP_K_REFINE, SURROGATE_TOP_K, PRUNE_RATIO, INVEST_STEP, DEFAULT_RISK_AVERSION:
  tuning knobs that govern shortlist size, pruning aggressiveness, investment granularity,
  and risk-averse scoring.

Outputs:
- Returns a Python list of dicts summarizing the investment policy (sequence).
- Writes artifacts:
    - results/ara_policy_sequence.csv
    - results/ara_policy_summary.json
"""
from typing import Any, List, Dict, Tuple, Optional
from pathlib import Path
import time
import json
import contextlib
import traceback
import os

import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed

# Helpers
from src.cascading_predictor import run_cascading_prediction, get_base_features_template, _reconcile_encoded_matrix
from src.fragility_mapper import CRITICAL_HUBS_TO_TEST

# Paths 
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTOR_MODEL_PATH = RESULTS_DIR / "best_model_final.joblib"

INVESTMENT_BUDGET = 5  # backwards-compatible constant 

# Tunable defaults (simple)
DEFAULT_MC_SAMPLES = 2
DEFAULT_SHOCK_NOISE_FRAC = 0.05
DEFAULT_TOP_K_REFINE = 1
DEFAULT_INVEST_STEP = 0.30
DEFAULT_RISK_AVERSION = 0.5
DEFAULT_N_JOBS = max(1, min(4, (os.cpu_count() or 1) - 1))
SURROGATE_TOP_K = 3
PRUNE_RATIO = 0.25
EPS = 1e-12
MIN_TRIALS_FOR_STD = 2


# Utility
def _quiet_call(func, *args, **kwargs):
    """Call func while suppressing stdout/stderr for tidy logs."""
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        return func(*args, **kwargs)


def _now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


# Matrix preparation
def _prepare_encoded_matrix(df_panel: pd.DataFrame, preprocessor: Any, feature_names_cleaned: List[str]):
    """Prepare encoded matrix and load persisted model. Returns model, X_encoded, encoded_names, df_meta, event_indices, X_original_events"""
    df_raw, df_meta, relevant_features = get_base_features_template(df_panel)
    expected_input_features = getattr(preprocessor, "input_feature_names_", relevant_features)
    df_align = df_raw[[c for c in expected_input_features if c in df_raw.columns]].copy()
    X_encoded = preprocessor.transform(df_align)
    encoded_names = list(preprocessor.get_feature_names_out())

    if not PREDICTOR_MODEL_PATH.exists():
        raise FileNotFoundError(f"Predictor model not found at {PREDICTOR_MODEL_PATH}")

    model = joblib.load(PREDICTOR_MODEL_PATH)
    model_expected_n = getattr(model, "n_features_in_", X_encoded.shape[1])
    X_rec, encoded_names = _reconcile_encoded_matrix(X_encoded, encoded_names, model_expected_n, feature_names_cleaned)

    # Identify event-like columns
    event_indices = [i for i, n in enumerate(encoded_names) if "events__n_events" in n]
    X_original_events = X_rec[:, event_indices].copy() if event_indices else np.zeros((X_rec.shape[0], 0), dtype=float)

    return model, X_rec, encoded_names, df_meta, event_indices, X_original_events


# Simulator wrapper & parsing
def _simulate_per_port(model: Any, X_encoded: np.ndarray, df_meta: pd.DataFrame, encoded_names: List[str], source: str, shock: float):
    """Run cascading prediction quietly, return mapping port->risk numeric."""
    results = _quiet_call(run_cascading_prediction,
                          model=model,
                          X_encoded=X_encoded,
                          df_meta=df_meta,
                          disrupted_port_name=source,
                          encoded_feature_names=encoded_names,
                          shock_magnitude=shock)
    if not results:
        return {}

    dfr = pd.DataFrame(results)

    # Detect numeric column containing risk-like values
    candidate_cols = ["Cascade_Risk", "Risk_Increase", "Risk_Increase_Numeric", "risk", "risk_increase"]
    risk_col = next((c for c in candidate_cols if c in dfr.columns), None)

    if risk_col is None:
        # fallback: first numeric column other than indices
        numeric_cols = [c for c in dfr.columns if pd.api.types.is_numeric_dtype(dfr[c])]
        if numeric_cols:
            risk_col = numeric_cols[0]
        else:
            return {}

    # Normalize to numeric
    dfr["__risk_num__"] = pd.to_numeric(dfr[risk_col], errors="coerce").fillna(0.0)

    # Determine port identifier column
    port_cols = ['Port', 'port', 'join_key', 'Join_Key', 'JoinKey', 'joinKey', 'Name', 'name']
    port_col = next((c for c in port_cols if c in dfr.columns), None)

    perport = {}
    if port_col is None:
        # use row index
        for idx, row in dfr.iterrows():
            perport[str(idx)] = perport.get(str(idx), 0.0) + float(row["__risk_num__"])
    else:
        for _, row in dfr.iterrows():
            p = row.get(port_col)
            if pd.isna(p):
                continue
            perport[str(p)] = perport.get(str(p), 0.0) + float(row["__risk_num__"])
    return perport


def _total_from_perport(perport: dict) -> float:
    return float(sum(perport.values())) if perport else 0.0


# CRN shock generator
def _generate_crn_shocks(samples: int, shock_noise_frac: float, seed: Optional[int] = None) -> List[Dict[str, float]]:
    samples = max(MIN_TRIALS_FOR_STD, int(samples))
    base = {c['source_port']: c['shock'] for c in CRITICAL_HUBS_TO_TEST}
    rng = np.random.default_rng(seed)
    trials = []
    for _ in range(samples):
        trial = {}
        for s, b in base.items():
            trial[s] = float(max(0.0, b + rng.normal(0.0, b * float(shock_noise_frac))))
        trials.append(trial)
    return trials

def _evaluate_cumulative_map(
    model: Any,
    X_base: np.ndarray,
    X_orig_events: np.ndarray,
    event_indices: List[int],
    df_meta: pd.DataFrame,
    encoded_names: List[str],
    cumulative_map: Dict[str, float],
    shocks: List[Dict[str, float]],
) -> Tuple[float, float]:
    """
    Compute mean and std of total risk under cumulative_map across shocks list.
    """
    totals = []
    for trial in shocks:
        X_mod = X_base.copy()
        # apply cumulative reductions relative to original event magnitudes
        for hub, frac in cumulative_map.items():
            # find matching row index for hub in df_meta
            jk = str(hub).strip().lower()
            # prefer exact join_key match if present
            matches = df_meta[df_meta["join_key"].astype(str).str.strip().str.lower() == jk]
            if matches.empty:
                # fallback to contains
                matches = df_meta[df_meta["join_key"].astype(str).str.contains(str(hub), case=False, na=False)]
            if matches.empty:
                continue
            row_idx = int(matches.index[0])
            for j, ei in enumerate(event_indices):
                orig_val = X_orig_events[row_idx, j]
                X_mod[row_idx, ei] = float(orig_val) * max(0.0, 1.0 - float(frac))

        trial_total = 0.0
        for src, shock_val in trial.items():
            perport = _simulate_per_port(model, X_mod, df_meta, encoded_names, src, shock_val)
            trial_total += _total_from_perport(perport)
        totals.append(trial_total)

    arr = np.array(totals, dtype=float)
    mean = float(arr.mean()) if arr.size > 0 else 0.0
    std = float(arr.std(ddof=0)) if arr.size > 1 else 0.0
    return mean, std


# Surrogate pre-score 
def _surrogate_scores(df_meta: pd.DataFrame, X_original_events: np.ndarray, event_indices: List[int]) -> Dict[str, float]:
    n = len(df_meta)
    event_sum = X_original_events.sum(axis=1) if event_indices and X_original_events.size > 0 else np.zeros(n)
    pagerank_cols = [c for c in df_meta.columns if 'pagerank' in c.lower()]
    pr = np.zeros(n)
    if pagerank_cols:
        try:
            pr = pd.to_numeric(df_meta[pagerank_cols[0]].fillna(0)).to_numpy(dtype=float)
        except Exception:
            pr = np.zeros(n)
    def norm(v):
        v = np.asarray(v, dtype=float)
        lo, hi = v.min(), v.max()
        if hi - lo <= EPS:
            return np.zeros_like(v)
        return (v - lo) / (hi - lo)
    prn = norm(pr); evn = norm(event_sum)
    scores = {}
    for i, row in df_meta.iterrows():
        key = str(row.get('join_key') or row.get('Port') or row.get('name') or i)
        scores[key] = float(0.6 * prn[i] + 0.4 * evn[i])
    return scores


def _policy_confidence(expected: float, std: float) -> float:
    if expected <= 0:
        return 0.0
    if std <= 0:
        return float(expected * 100.0)
    return float(expected / (std + EPS))


# Main run_ara_agent
def run_ara_agent(
    df_panel: pd.DataFrame,
    feature_names_cleaned: List[str],
    preprocessor: Any,
    investment_budget: int = INVESTMENT_BUDGET,
    mc_samples: int = DEFAULT_MC_SAMPLES,
    shock_noise_frac: float = DEFAULT_SHOCK_NOISE_FRAC,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    n_jobs: int = DEFAULT_N_JOBS,
    top_k_refine: int = DEFAULT_TOP_K_REFINE,
    invest_step: float = DEFAULT_INVEST_STEP,
    surrogate_k: int = SURROGATE_TOP_K,
    seed: Optional[int] = 42,
    fast_mode: bool = False,
    save_artifacts: bool = True,
    verbose: bool = True,
    **kwargs
) -> List[Dict]:
    """
    Returns a sequence list of dicts describing chosen investments.
    """
    t0 = time.time()
    if verbose:
        print("\n" + "-" * 100)
        print("ARA: Monte-Carlo Investment Finder - Start")
        print("-" * 100)

    # Prepare model & encoded matrix
    try:
        model, X_encoded, encoded_names, df_meta, event_indices, X_original_events = _prepare_encoded_matrix(df_panel, preprocessor, feature_names_cleaned)
    except Exception as e:
        print(f"  (x) ARA setup failed: {e}")
        traceback.print_exc()
        return []

    # Working copy and snapshot
    X_work = X_encoded.copy()
    X_snapshot = X_encoded.copy()

    if fast_mode:
        mc_samples = max(1, int(mc_samples))
        n_jobs = 1

    shocks = _generate_crn_shocks(mc_samples, shock_noise_frac, seed=seed)

    # Surrogate scoring 
    surrogate_scores = _surrogate_scores(df_meta, X_original_events, event_indices)

    # Baseline
    baseline_mean, baseline_std = _evaluate_cumulative_map(model, X_snapshot, X_original_events, event_indices, df_meta, encoded_names, {}, shocks)
    baseline_std_str = f"{baseline_std:.6f}" if baseline_std and baseline_std > 0 else "N/A"
    if verbose:
        print(f"  Baseline systemic risk (mean ± std): {baseline_mean:.6f} ± {baseline_std_str}\n")

    candidates = [d['source_port'] for d in CRITICAL_HUBS_TO_TEST]
    sequence: List[Dict] = []
    cumulative_map: Dict[str, float] = {}

    for step in range(1, int(investment_budget) + 1):
        if not candidates:
            if verbose:
                print("  No candidates remaining; stopping.")
            break

        current_remaining = sequence[-1]['Cumulative_Risk_Remaining'] if sequence else baseline_mean

        # Surrogate shortlist
        ranked = sorted(candidates, key=lambda c: surrogate_scores.get(c, 0.0), reverse=True)
        pre_candidates = ranked[:max(surrogate_k, len(ranked))]
        # Prune further by top-K surrogate if many candidates
        pre_candidates = pre_candidates[:max(6, surrogate_k)]

        # Coarse (parallel) evaluation of pre_candidates
        def eval_candidate(cand):
            hyp = dict(cumulative_map)
            hyp[cand] = min(0.99, hyp.get(cand, 0.0) + invest_step)
            mean_after, std_after = _evaluate_cumulative_map(model, X_snapshot, X_original_events, event_indices, df_meta, encoded_names, hyp, shocks)
            return cand, mean_after, std_after

        if n_jobs <= 1:
            coarse_results = [eval_candidate(c) for c in pre_candidates]
        else:
            coarse_results = Parallel(n_jobs=n_jobs)(delayed(eval_candidate)(c) for c in pre_candidates)

        # Compute reductions and prune
        reductions = [(current_remaining - m, c, m, s) for (c, m, s) in coarse_results]
        best_coarse = max([r[0] for r in reductions]) if reductions else 0.0
        prune_thresh = PRUNE_RATIO * best_coarse if best_coarse > 0 else 0.0
        pruned = [r for r in reductions if r[0] >= prune_thresh]
        pruned_sorted = sorted(pruned, key=lambda x: x[0], reverse=True)
        shortlist = [r[1] for r in pruned_sorted[:top_k_refine]] if pruned_sorted else []

        if verbose:
            print(f"Step {step}: pre-candidates={len(pre_candidates)}, shortlisted={shortlist if shortlist else 'None'} (prune>= {prune_thresh:.6f})")

        if not shortlist:
            if verbose:
                print("  No promising candidates after pruning. Ending.")
            break

        # Refined eval on shortlist 
        refined_shocks = shocks  # using same shocks
        def eval_refined(cand):
            hyp = dict(cumulative_map)
            hyp[cand] = min(0.99, hyp.get(cand, 0.0) + invest_step)
            mean_after, std_after = _evaluate_cumulative_map(model, X_snapshot, X_original_events, event_indices, df_meta, encoded_names, hyp, refined_shocks)
            return cand, mean_after, std_after

        if n_jobs <= 1:
            refined_results = [eval_refined(c) for c in shortlist]
        else:
            refined_results = Parallel(n_jobs=n_jobs)(delayed(eval_refined)(c) for c in shortlist)

        # Pick best by expected reduction minus risk_aversion * std
        best_score = -1e18
        best_choice = None
        for cand, mean_after, std_after in refined_results:
            expected = current_remaining - mean_after
            std_val = std_after if std_after is not None else 0.0
            score = expected - risk_aversion * std_val
            if score > best_score:
                best_score = score
                best_choice = (cand, expected, mean_after, std_val, score)

        if best_choice is None or best_choice[1] <= 0:
            if verbose:
                print("  No beneficial candidate found after refinement. Ending.")
            break

        cand, expected_reduction, mean_after, std_after, score = best_choice
        if verbose:
            std_str = f"{std_after:.6f}" if std_after and std_after > 0 else "N/A"
            print(f"  DECISION -> Invest in {cand} | Expected Δrisk={expected_reduction:.6f} | est.std={std_str} | score={score:.6f}")

        # Commit investment (update cumulative_map and mutate X_work)
        prev = cumulative_map.get(cand, 0.0)
        new_frac = min(0.99, prev + invest_step)
        cumulative_map[cand] = new_frac

        # Apply to X_work (relative to original)
        matches = df_meta[df_meta["join_key"].astype(str).str.strip().str.lower() == str(cand).strip().lower()]
        if matches.empty:
            matches = df_meta[df_meta["join_key"].astype(str).str.contains(str(cand), case=False, na=False)]
        if not matches.empty:
            row_idx = int(matches.index[0])
            for j, ei in enumerate(event_indices):
                orig_val = X_original_events[row_idx, j]
                X_work[row_idx, ei] = float(orig_val) * max(0.0, 1.0 - float(new_frac))
        else:
            if verbose:
                print(f"   (warn) Could not locate metadata row for {cand}; investment applied logically but not persisted in X_work.")

        # Verify outcome on committed X_work
        mean_committed, std_committed = _evaluate_cumulative_map(model, X_work, X_original_events, event_indices, df_meta, encoded_names, cumulative_map, shocks)

        policy_conf = _policy_confidence(expected_reduction, std_committed)

        sequence.append({
            "Step": step,
            "Invest_Hub": cand,
            "Risk_Reduction_Marginal": float(expected_reduction),
            "Cumulative_Risk_Remaining": float(mean_committed),
            "Cumulative_Reduction_Fraction": float(new_frac),
            "Policy_Confidence": float(policy_conf)
        })

        # Remove chosen candidate
        if cand in candidates:
            candidates.remove(cand)

        # Early stop
        if mean_committed <= 0.0:
            if verbose:
                print("  Systemic risk effectively eliminated; stopping early.")
            break

        if verbose:
            print("")

    # Persist artifacts output
    if save_artifacts:
        try:
            df_seq = pd.DataFrame(sequence)
            seq_csv = RESULTS_DIR / "ara_policy_sequence.csv"
            df_seq.to_csv(seq_csv, index=False)
            summary = {"timestamp": time.time(), "investment_budget": investment_budget, "sequence": sequence}
            with open(RESULTS_DIR / "ara_policy_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            if verbose:
                print(f"  (v) Saved sequence -> {seq_csv}")
        except Exception:
            traceback.print_exc()

    if verbose:
        if sequence:
            print("\nFinal ARA sequence:")
            df_out = pd.DataFrame(sequence)
            display_cols = ["Step", "Invest_Hub", "Risk_Reduction_Marginal", "Cumulative_Risk_Remaining", "Cumulative_Reduction_Fraction", "Policy_Confidence"]
            print(df_out[display_cols].to_string(index=False, float_format="%.6f"))
        else:
            print("No investments recommended by ARA.")
        print(f"\nARA elapsed time: {time.time() - t0:.1f}s")

    return sequence