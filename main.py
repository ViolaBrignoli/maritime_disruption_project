"""
MARITIME DISRUPTION PROJECT - Main Execution Entry Point

This script is the reproducible, end-to-end runner for the Maritime Disruption
project. It enforces deterministic behavior up front (seed + stable filesystem
ordering) and then executes the full pipeline in the following stages:

  Determinism setup
     - Sets PYTHONHASHSEED and MARITIME_SEED, seeds Python and NumPy RNGs,
       and applies sorted file-listing/glob monkey-patches so file discovery
       is stable across runs and platforms.

  Data preparation (ETL)
     - Runs the data pipeline (src.data_prep.run_data_pipeline) to create
       standardized CSVs under data/processed if they are missing.

  Network & panel preparation
     - Generates network centrality features (src.network_graph) if missing.
     - Builds the port-year panel and fits/returns a preprocessing pipeline
       (src.data_loader.load_and_split_panel).

  Baselines and model training
     - Runs a simple Markov baseline.
     - Trains and compares ML models (src.models.train_all_models),
       selects/exports the best model artifact and xgb.

  Evaluation & diagnostics
     - Produces evaluation reports, calibration/PR/ROC artifacts (src.evaluation).
     - Computes a mutual-information Top-K feature snapshot for cascade analysis.

  Predictive validation (cascading stress tests)
     - Runs batch cascading simulations (src.cascading_predictor) for scenario
       stress tests and saves consolidated results and cumulative risk plots.

  Structural analyses & prescriptions
     - Fragility mapping (src.fragility_mapper) to produce quadrant diagnostics.
     - Economic Value-of-Resilience investment optimization (src.investment_optimizer).

  Autonomous Resilience Agent (ARA)
     - Runs the ARA policy search to produce sequential investment recommendations and persists the policy.

Outputs & reproducibility notes
  - All artifacts are written to results/ (models, CSVs, PNGs).
  - The RANDOM_SEED set at the top controls reproducibility

Usage
  - Run: python main.py
"""
import sys
import os
import warnings
import traceback
from pathlib import Path

# Determinism 
RANDOM_SEED = 42  

# Set environment seeds early
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
os.environ["MARITIME_SEED"] = str(RANDOM_SEED)

# Seed Python & NumPy RNGs
import random as _random
_random.seed(RANDOM_SEED)
import numpy as _np
_np.random.seed(RANDOM_SEED)

# Monkey-patch filesystem listing and pathlib.Path.glob to be deterministic.
import os as _os
import glob as _glob
import pathlib as _pathlib

# Save originals
_orig_listdir = _os.listdir
_orig_glob = _glob.glob
_orig_pathlib_glob = _pathlib.Path.glob

def _sorted_listdir(path):
    try:
        items = _orig_listdir(path)
    except Exception:
        # fall back to original to raise original error
        return _orig_listdir(path)
    return sorted(items, key=lambda x: str(x))

def _sorted_glob(pattern, recursive=False):
    try:
        res = _orig_glob(pattern, recursive=recursive)
    except Exception:
        return _orig_glob(pattern, recursive=recursive)
    return sorted(res, key=lambda x: str(x))

def _pathlib_sorted_glob(self, pattern):
    # original returns generator; we return iterator over sorted list
    try:
        res = list(_orig_pathlib_glob(self, pattern))
    except Exception:
        return _orig_pathlib_glob(self, pattern)
    res.sort(key=lambda p: str(p))
    return iter(res)

# Apply monkeypatches
_os.listdir = _sorted_listdir
_glob.glob = _sorted_glob
_pathlib.Path.glob = _pathlib_sorted_glob

# Import everything after patching to ensure deterministic behavior in imports/ETL
# Ignore warnings for cleaner terminal visual
warnings.filterwarnings("ignore")

# Add src to sys.path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Standard imports used by the main pipeline 
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

# Ensure results dir exists
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Now import project modules
try:
    from src.data_prep import run_data_pipeline
    from src.network_graph import run_network_analysis
    from src.data_loader import load_and_split_panel, load_all_processed_data
    from src.models import train_all_models
    from src.evaluation import generate_comprehensive_report
    from src.markov import run_markov_benchmark_panel
    from src.cascading_predictor import execute_batch_stress_test
    from stress_tester import plot_stress_test_report
    from src.fragility_mapper import execute_fragility_analysis
    from src.investment_optimizer import execute_investment_optimization
    # ARA import is conditional later (for separation)
except Exception as e:
    print(f"\n[CRITICAL ERROR] Could not import modules: {e}")
    traceback.print_exc()
    # Restore patched functions to originals before exit 
    _os.listdir = _orig_listdir
    _glob.glob = _orig_glob
    _pathlib.Path.glob = _orig_pathlib_glob
    sys.exit(1)

# Tunables 
TOPK_REPORT = 10
TOPK_SAVE = 40
MI_SAMPLE_FRAC = 0.25
MI_NEIGHBORS = 3
SAVE_TOPK_CSV = True
PREDICTOR_MODEL_PATH = RESULTS_DIR / "best_model_final.joblib"
RANDOM_STATE = RANDOM_SEED

DISRUPTION_SCENARIOS = [
    {"port": "Singapore", "shock": 5.0},
    {"port": "Suez", "shock": 6.0},
    {"port": "Shanghai", "shock": 8.0},
    {"port": "Rotterdam", "shock": 5.0},
    {"port": "Durban", "shock": 7.0},
    {"port": "Valparaiso", "shock": 6.0},
]

# MI-based Top-K feature snapshot (because all too heavy)
def _as_dataframe(X, feature_names=None):
    if X is None:
        return None
    if isinstance(X, pd.DataFrame):
        return X
    try:
        arr = np.asarray(X)
        if arr.ndim == 2:
            if feature_names is not None and len(feature_names) == arr.shape[1]:
                return pd.DataFrame(arr, columns=feature_names)
            return pd.DataFrame(arr)
    except Exception:
        pass
    return None

def compute_topk_features_mi(
    X_test_cleaned,
    y_test,
    feature_names_cleaned,
    topk_report=TOPK_REPORT,
    topk_save=TOPK_SAVE,
    sample_frac=MI_SAMPLE_FRAC,
    n_neighbors=MI_NEIGHBORS,
    out_dir=RESULTS_DIR,
):
    try:
        if X_test_cleaned is None or y_test is None or not feature_names_cleaned:
            print(" (i) Skipping MI Top-K: missing test data or feature names.")
            return pd.DataFrame(columns=["Rank", "Feature", "Importance"])

        X_df = _as_dataframe(X_test_cleaned, feature_names_cleaned)
        if X_df is None:
            try:
                X_df = pd.DataFrame(np.asarray(X_test_cleaned), columns=feature_names_cleaned[: np.asarray(X_test_cleaned).shape[1]])
            except Exception:
                print(" (!) Could not coerce X_test_cleaned to DataFrame for MI. Aborting MI snapshot.")
                return pd.DataFrame(columns=["Rank", "Feature", "Importance"])

        n_rows = X_df.shape[0]
        if n_rows > 2000 and 0.0 < sample_frac < 1.0:
            sample_n = max(200, int(n_rows * sample_frac))
            X_sample = X_df.sample(n=sample_n, random_state=RANDOM_STATE)
            if hasattr(y_test, "loc"):
                y_sample = y_test.loc[X_sample.index]
            else:
                y_arr = np.asarray(y_test)
                y_sample = y_arr[X_sample.index]
            print(f" (i) MI prefilter sampling: using {len(X_sample)} / {n_rows} rows (~{int(100*len(X_sample)/n_rows)}%).")
        else:
            X_sample = X_df
            y_sample = y_test if not hasattr(y_test, "loc") else y_test.loc[X_df.index]
            print(f" (i) MI prefilter using full test set: {n_rows} rows.")

        try:
            mi_vals = mutual_info_classif(
                X_sample.values,
                np.asarray(y_sample).ravel(),
                discrete_features=False,
                random_state=RANDOM_STATE,
                n_neighbors=n_neighbors,
            )
        except Exception as e_mi:
            print(f" (i) mutual_info_classif raised an error, falling back to absolute correlation: {e_mi}")
            yv = np.asarray(y_sample).ravel()
            mi_vals = np.array([
                abs(np.corrcoef(X_sample.iloc[:, i].fillna(0).values, yv)[0, 1]) if X_sample.shape[0] > 1 else 0.0
                for i in range(X_sample.shape[1])
            ])
            mi_vals = np.nan_to_num(mi_vals)

        feat_names = list(X_df.columns)
        mi_df = pd.DataFrame({"Feature": feat_names, "Importance": mi_vals})
        mi_df = mi_df.fillna(0.0).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        top_n_report = min(topk_report, mi_df.shape[0])
        df_top_report = mi_df.head(top_n_report).copy()
        df_top_report.insert(0, "Rank", range(1, len(df_top_report) + 1))

        top_n_save = min(topk_save, mi_df.shape[0])
        df_top_save = mi_df.head(top_n_save).copy()
        df_top_save.insert(0, "Rank", range(1, len(df_top_save) + 1))

        if SAVE_TOPK_CSV:
            try:
                out_path = out_dir / "top_features_mi_for_cascade_investigation.csv"
                df_top_save.to_csv(out_path, index=False)
                print(f" (v) Top-{top_n_save} feature snapshot saved to: {out_path}")
            except Exception as e:
                print(f" (!) Could not persist Top-{top_n_save} features CSV: {e}")

        return df_top_report[["Rank", "Feature", "Importance"]]

    except Exception as e:
        print(f" (x) compute_topk_features_mi failed: {e}")
        traceback.print_exc()
        return pd.DataFrame(columns=["Rank", "Feature", "Importance"])

# ---------------------------------------------------------------------

def main():
    
    X_train, X_test, y_train, y_test, df_panel, feature_names, preprocessor = [None] * 7
    models_dict = {}
    metrics_dict = {}
    best_model_name = "None"
    best_auc = -1
    best_model_obj = None
    feature_names_cleaned = []
    df_fragility_map = pd.DataFrame()
    df_chord_matrix = pd.DataFrame()
    all_results_raw = []
    df_ara_policy = pd.DataFrame()
    df_investment_report = pd.DataFrame()

    print("\n" + "=" * 100)
    print(" MARITIME NETWORK DISRUPTION PROJECT - EXECUTION PIPELINE ")
    print("=" * 100)

    # ---------------------------------------------------------------------
    # STEP 1: DATA PREPARATION (ETL)
    # ---------------------------------------------------------------------
    print("\n[STEP 1] Checking Data Pipeline Status...")
    processed_dir = Path("data/processed")

    if not processed_dir.exists() or not any(processed_dir.iterdir()):
        print(" (!) Processed data not found. Initiating ETL Pipeline...")
        try:
            # run_data_pipeline will now see deterministic filesystem order & seeded RNGs
            run_data_pipeline()
            print(" (v) Data Preparation Complete.")
        except Exception as e:
            print(f" (x) Critical Error during Data Prep: {e}")
            traceback.print_exc()
            # Restore patched functions to originals before exit
            _os.listdir = _orig_listdir
            _glob.glob = _orig_glob
            _pathlib.Path.glob = _orig_pathlib_glob
            sys.exit(1)
    else:
        print(" (v) Processed data folder found. Skipping raw data processing.")
    datasets = load_all_processed_data()

    # ---------------------------------------------------------------------
    # STEP 1.5: NETWORK FEATURE ENGINEERING
    # ---------------------------------------------------------------------
    print("\n[STEP 1.5] Checking Network Graph Features...")
    network_check = Path("data/processed/df_Network_Centrality_Features.csv")

    if not network_check.exists():
        print(" (!) Network features not found. Generating Graph Metrics...")
        try:
            run_network_analysis()
            print(" (v) Network features generated.")
            datasets = load_all_processed_data()
            print(" (i) Processed data dictionary successfully updated with new network features.")
        except Exception as e:
            print(f" (x) Network analysis failed: {e}")
    else:
        print(" (v) Network features found. Skipping calculation.")

    # ---------------------------------------------------------------------
    # STEP 2: DATA LOADING & SPLITTING
    # ---------------------------------------------------------------------
    print("\n[STEP 2] Loading and Preprocessing Panel Data...")
    try:
        loaded_data = load_and_split_panel(year_cutoff=2020)
        if len(loaded_data) == 7:
            X_train, X_test, y_train, y_test, df_panel, feature_names, preprocessor = loaded_data
        else:
            X_train, X_test, y_train, y_test, df_panel, feature_names = loaded_data
            preprocessor = None
            print(" [!] Warning: Data loader did not return preprocessor.")

        print(" (v) Panel Data Loaded Successfully.")
        print(f" - Training Samples: {X_train.shape[0]}")
        print(f" - Test Samples: {X_test.shape[0]}")
        print(f" - Feature Count: {X_train.shape[1]}")

    except Exception as e:
        print(f" (x) Error loading data: {e}")
        traceback.print_exc()
        # Restore patched functions to originals before exit
        _os.listdir = _orig_listdir
        _glob.glob = _orig_glob
        _pathlib.Path.glob = _orig_pathlib_glob
        sys.exit(1)

    # ---------------------------------------------------------------------
    # STEP 3: MODEL TRAINING
    # ---------------------------------------------------------------------
    print("\n[STEP 3] Model Training & Cross-Validation...")

    # Markov baseline
    print(f"> Running Markov Chain Baseline...")
    markov_results = {}
    try:
        if df_panel is None:
            raise RuntimeError("df_panel is not available for Markov baseline (skipping).")

        year_cutoff = 2020
        df_train_panel = df_panel[df_panel["year"] <= year_cutoff].copy()
        df_test_panel = df_panel[df_panel["year"] >= (year_cutoff + 1)].copy()

        markov_results = run_markov_benchmark_panel(
            df_train=df_train_panel,
            df_test=df_test_panel,
            group_col="join_key",
            time_col="year",
            target_col="target_is_disrupted_in_year",
            laplace=1.0,
            use_observed_prev=False,
            verbose=True,
        )

        print(f" (i) MARKOV BASELINE: Accuracy={markov_results.get('accuracy', 0.0):.4f}  F1={markov_results.get('f1', 0.0):.4f}")
    except Exception as e:
        print(f" (x) Markov baseline execution failed: {e}")
        pass

    # Train ML models
    try:
        print("\n" + "=" * 100)
        print("TRAINING MACHINE LEARNING MODELS")
        print("=" * 100)
        results, X_train_cleaned, X_test_cleaned, feature_names_cleaned = train_all_models(
            X_train, y_train, X_val=X_test, y_val=y_test, include_xgb=True, feature_names=feature_names
        )
        models_dict = results.get("models", {})
        metrics_dict = results.get("metrics", {})

        best_auc = -1
        for name, met in metrics_dict.items():
            if isinstance(met, dict) and "val" in met:
                auc = met["val"].get("roc_auc", 0.0)
                if auc > best_auc:
                    best_auc = auc
                    best_model_name = name
                    best_model_obj = models_dict[name]

        # MI Top-K snapshot (not all because too heavy)
        try:
            df_top_mi = compute_topk_features_mi(
                X_test_cleaned,
                y_test,
                feature_names_cleaned,
                topk_report=TOPK_REPORT,
                topk_save=TOPK_SAVE,
                sample_frac=MI_SAMPLE_FRAC,
                n_neighbors=MI_NEIGHBORS,
                out_dir=RESULTS_DIR,
            )
            if not df_top_mi.empty:
                pd.set_option("display.width", 200)
                print("\n" + "-" * 100)
                print(f"TOP {min(TOPK_REPORT, df_top_mi.shape[0])} MOST PROMINENT FEATURES (MI SNAPSHOT)")
                print("-" * 100)
                print(df_top_mi.to_string(index=False, float_format="%.4f"))
                pd.reset_option("display.width")
        except Exception as e:
            print(f" (x) Failed to compute MI Top-K: {e}")
            traceback.print_exc()
            # Restore patched functions
            _os.listdir = _orig_listdir
            _glob.glob = _orig_glob
            _pathlib.Path.glob = _orig_pathlib_glob
            sys.exit(1)

    except Exception as e:
        print(f" (x) Training Failed: {e}")
        traceback.print_exc()
        # Restore patched functions
        _os.listdir = _orig_listdir
        _glob.glob = _orig_glob
        _pathlib.Path.glob = _orig_pathlib_glob
        sys.exit(1)

    # ---------------------------------------------------------------------
    # STEP 4: EVALUATION AND BEST MODEL PERSISTENCE
    # ---------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("STEP 4: EVALUATION AND BEST MODEL PERSISTENCE")
    print("=" * 100)
    if models_dict:
        evaluation_summary = generate_comprehensive_report(models_dict, X_test_cleaned, y_test, feature_names=feature_names_cleaned)
        print("\n" + "-" * 100)
        print("BEST MODEL ANNOUNCEMENT")
        print("-" * 100)
        print(f" (v) Primary Simulation Model Selected: {best_model_name.upper()}")
        print(f" (i) Performance (Test AUC): {best_auc:.4f}")
        try:
            if best_model_obj:
                joblib.dump(best_model_obj, PREDICTOR_MODEL_PATH)
                print(f" (v) Best model '{best_model_name}' persisted for downstream use.")
        except Exception as e:
            print(f" (!) Error persisting best model: {e}")
    else:
        print(" (!) No models available to evaluate or persist.")


    # ---------------------------------------------------------------------
    # STEP 5: CASCADING SIMULATION STRESS TEST
    # ---------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("STEP 5: PREDICTIVE VALIDATION - CASCADING SIMULATION STRESS TEST")
    print("=" * 100)
    if best_model_name != "None" and preprocessor is not None and df_panel is not None:
        try:
            all_results_raw = execute_batch_stress_test(df_panel, feature_names_cleaned, preprocessor, DISRUPTION_SCENARIOS)
            if all_results_raw:
                df_all = pd.DataFrame(all_results_raw)
                df_all["Risk_Increase_Numeric"] = pd.to_numeric(df_all["Risk_Increase"], errors="coerce")
                df_all["XGB_Risk_Prob_Numeric"] = pd.to_numeric(df_all["XGB_Risk_Prob"], errors="coerce")
                df_filtered = df_all[(df_all["Risk_Increase_Numeric"] > 0.0000)]
                df_final_report = df_filtered.sort_values(by=["Risk_Increase_Numeric", "XGB_Risk_Prob_Numeric"], ascending=False)
                print("\n" + "-" * 100)
                print(f"VALIDATION REPORT: ALL {len(df_final_report)} AFFECTED PORTS (EVIDENCE BASE)")
                print("-" * 100)
                pd.set_option("display.max_rows", None)
                pd.set_option("display.max_columns", None)
                pd.set_option("display.width", 1000)
                print(df_final_report[["Port", "Scenario_Source", "Confidence_Tier", "XGB_Risk_Prob", "Risk_Increase"]].to_string())
                pd.reset_option("display.max_rows")
                pd.reset_option("display.max_columns")
                pd.reset_option("display.width")
                plot_stress_test_report(df_final_report)
        except Exception as e:
            print(f" (x) Stress Test Execution Failed: {e}")
            traceback.print_exc()
            all_results_raw = []
    else:
        print("\n[STEP 5] Stress Test skipped due to missing models or data.")

    # ---------------------------------------------------------------------
    # STEP 6: STRUCTURAL RESILIENCE: FRAGILITY MAPPING
    # ---------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("STEP 6: STRUCTURAL RESILIENCE: FRAGILITY MAPPING (Data & Artifacts)")
    print("=" * 100)
    if df_panel is not None and preprocessor is not None and all_results_raw:
        try:
            df_fragility_map = execute_fragility_analysis(df_panel, feature_names_cleaned, preprocessor, all_results_raw)
            print("(v) Fragility Analysis completed.")
        except Exception as e:
            print(f" (x) Fragility Mapping Failed: {e}")
            traceback.print_exc()
            df_fragility_map = pd.DataFrame()
            df_chord_matrix = pd.DataFrame()
    else:
        print(" (i) Fragility Mapping skipped: Missing panel data, preprocessor, or stress test results.")

    # ---------------------------------------------------------------------
    # STEP 7: SCENARIO VULNERABILITY RANKING
    # ---------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("STEP 7: SCENARIO VULNERABILITY RANKING (Systemic Threat)")
    print("=" * 100)
    if df_fragility_map is not None and not df_fragility_map.empty:
        df_scenario_rank = df_fragility_map[["Source_Hub", "Intensity (Ip)", "Fragility_Index", "Shock_Magnitude"]].sort_values(by="Intensity (Ip)", ascending=False).reset_index(drop=True)
        df_scenario_rank["Threat_Rank"] = df_scenario_rank.index + 1
        df_output = df_scenario_rank[["Threat_Rank", "Source_Hub", "Intensity (Ip)", "Fragility_Index", "Shock_Magnitude"]]
        print("\n" + "-" * 100)
        print("GLOBAL THREAT RANKING (Scenarios Ranked by Total Systemic Risk Caused)")
        print("-" * 100)
        print("The Initial Intensity (Ip) measures the total predicted risk generated by the shock.")
        print(df_output.to_string(index=False, float_format="%.4f"))
    else:
        print(" (i) Scenario Vulnerability Ranking skipped: Fragility Mapping data (Ip) not available.")

    # ---------------------------------------------------------------------
    # STEP 8: EVR OPTIMIZATION INVEESTMENT PORTFOLIO
    # ---------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("STEP 8: ECONOMIC PRESCRIPTION: EVR OPTIMIZATION")
    print("=" * 100)
    df_investment_report = pd.DataFrame()
    AUTO_SELECT_INVESTMENT = True
    if preprocessor is not None and df_panel is not None:
        try:
            df_investment_report = execute_investment_optimization(df_panel, feature_names_cleaned, preprocessor, auto_select=AUTO_SELECT_INVESTMENT)
        except Exception as e:
            print(f" (x) Investment Optimization Failed: {e}")
            traceback.print_exc()
            # Restore patched functions to originals before exit
            _os.listdir = _orig_listdir
            _glob.glob = _orig_glob
            _pathlib.Path.glob = _orig_pathlib_glob
            sys.exit(1)

    if df_investment_report is not None and not df_investment_report.empty:
        print("\n" + "-" * 100)
        print("OPTIMAL RESILIENCE INVESTMENT PORTFOLIO (Marginal ROI Ranking)")
        print("-" * 100)
        print(df_investment_report[["Investment_Rank", "Source_Hub", "Global_Risk_Reduction"]].to_string(index=False))
    else:
        print(" (i) Investment Optimization produced no results or was skipped.")

    # ---------------------------------------------------------------------
    # STEP 9: ARA POLICY OPTIMIZATION
    # ---------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("STEP 9: AUTONOMOUS RESILIENCE AGENT (ARA) - POLICY OPTIMIZATION")
    print("=" * 100)

    run_ara_agent = None
    try:
        from src.ara_agent2 import run_ara_agent
        imported_from = "src.ara_agent2"
    except Exception:
        try:
            from src.ara_agent import run_ara_agent
            imported_from = "src.ara_agent"
        except Exception:
            try:
                from src.autonomous_agent import run_ara_agent
                imported_from = "src.autonomous_agent"
            except Exception:
                run_ara_agent = None
                imported_from = None

    if imported_from:
        print(f" (i) Using ARA implementation imported from: {imported_from}")

    if df_panel is not None and preprocessor is not None and run_ara_agent is not None:
        try:
            if not PREDICTOR_MODEL_PATH.exists():
                print(" (x) ERROR: Primary model (best_model_final.joblib) not found. Cannot run ARA.")
                df_ara_policy = pd.DataFrame()
            else:
                ara_results = run_ara_agent(df_panel, feature_names_cleaned, preprocessor)
                df_ara_policy = pd.DataFrame(ara_results)
        except Exception as e:
            print(f" (x) ARA Optimization Failed: {e}")
            traceback.print_exc()
            df_ara_policy = pd.DataFrame()
    else:
        if run_ara_agent is None:
            print(" (x) ARA implementation not found (tried src.ara_agent2, src.ara_agent, src.autonomous_agent). Skipping ARA.")
        df_ara_policy = pd.DataFrame()


    # ---------------------------------------------------------------------
    # PIPELINE COMPLETED
    # ---------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("PIPELINE COMPLETED SUCCESSFULLY")

    # Restore patched functions back to originals at the end of run
    _os.listdir = _orig_listdir
    _glob.glob = _orig_glob
    _pathlib.Path.glob = _orig_pathlib_glob

if __name__ == "__main__":
    main()