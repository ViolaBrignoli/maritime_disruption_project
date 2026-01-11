"""
MARITIME DISRUPTION PROJECT - Investment Optimization (Economic Value of Resilience)

Implements an Economic Value of Resilience (EVR) optimizer used to propose and evaluate port-level resilience investments
via cascading disruption simulations.
Orchestrates per-hub investment recommendation and evaluation using the
persisted predictive cascade simulator. The module computes baseline systemic
risk for a small set of critical hub scenarios, auto-selects sensible
reduction fractions (or maps investments to reductions), simulates marginal
impacts (ΔI_p) of those investments, ranks candidate investments by marginal
EVR and ROI, persists results, and produces a simple ranking visualization.

- Internal investment generation:
    * Auto-selects per-hub reduction fractions using a small elbow-based
      heuristic over a reduction grid when auto_select is enabled.
    * Converts reduction fractions ⇄ investment units using a saturating
      exponential mapping and its inverse (analytical).
    * Generates the investment_map internally 

- Cascade simulation & metrics:
    * Runs quiet cascade simulations to compute baseline Ip and post-investment Ip for each hub.
    * Reports per-hub: Initial_Ip, Reduced_Ip, Global_Risk_Reduction (ΔI_p),
      Selected_Reduction_Fraction, Investment_Amount, and ROI per investment unit.
    * If a suggested investment produces no detectable benefit, the module
      escalates investment within configured caps to find the minimal
      detectable improvement (preserving prior behavior).

- Outputs & persistence:
    * Persists recommendations to results/investment_recommendations.csv and
      results/investment_pareto_frontier.csv.
    * Produces a horizontal barplot saved as results/resilience_investment_ranking.png.
    * Returns a compact DataFrame with the final ranking and metrics.
"""
from typing import Any, List, Dict, Tuple, Optional
from pathlib import Path
import math
import traceback
import contextlib
import os

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from src.cascading_predictor import run_cascading_prediction, PREDICTOR_MODEL_PATH
from src.cascading_predictor import get_base_features_template, _reconcile_encoded_matrix

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Default hub scenarios
CRITICAL_HUBS_TO_TEST = [
    {"source_port": "Singapore", "shock": 5.0},
    {"source_port": "Shanghai", "shock": 8.0},
    {"source_port": "Rotterdam", "shock": 5.0},
    {"source_port": "Suez", "shock": 6.0},
    {"source_port": "Durban", "shock": 7.0},
    {"source_port": "Valparaiso", "shock": 6.0},
]

# Defaults used in the pipeline and preserved to match existing outputs
DEFAULT_INVESTMENT_SCALE = 1.0
DEFAULT_MAX_REDUCTION = 0.30
RESILIENCE_IMPROVEMENT_FACTOR = 0.10
GRID_POINTS = 10
ELBOW_THRESHOLD_RATIO = 0.25

# Helpers
def calculate_shock_intensity(
    model: Any,
    X_encoded_base: np.ndarray,
    df_meta: pd.DataFrame,
    encoded_names: List[str],
    source_port: str,
    shock_magnitude: float,
    quiet: bool = True
) -> Tuple[float, int]:
    """Run a single cascade and return (Ip, affected_count). Quiet."""
    try:
        if quiet:
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                all_results = run_cascading_prediction(
                    model=model,
                    X_encoded=X_encoded_base,
                    df_meta=df_meta,
                    disrupted_port_name=source_port,
                    encoded_feature_names=encoded_names,
                    shock_magnitude=shock_magnitude
                )
        else:
            all_results = run_cascading_prediction(
                model=model,
                X_encoded=X_encoded_base,
                df_meta=df_meta,
                disrupted_port_name=source_port,
                encoded_feature_names=encoded_names,
                shock_magnitude=shock_magnitude
            )
    except Exception:
        return 0.0, 0

    if not all_results:
        return 0.0, 0

    df = pd.DataFrame(all_results)
    # Robustly derive numeric risk column (fall back to Risk_Increase or first numeric)
    if 'Risk_Increase' in df.columns:
        vals = pd.to_numeric(df['Risk_Increase'], errors='coerce').fillna(0.0)
    else:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        vals = pd.to_numeric(df[num_cols[0]], errors='coerce').fillna(0.0) if num_cols else pd.Series([0.0])
    Ip = float(vals.sum())
    return Ip, int(len(df))


def _investment_to_reduction(investment: float, investment_scale: float, max_reduction: float) -> float:
    if investment <= 0:
        return 0.0
    reduction = float(max_reduction) * (1.0 - np.exp(-float(investment) / float(investment_scale)))
    return min(max(reduction, 0.0), float(max_reduction))


def _reduction_to_investment(reduction: float, investment_scale: float, max_reduction: float) -> float:
    r = min(float(reduction), max_reduction * 0.999999)
    if r <= 0:
        return 0.0
    inv = -float(investment_scale) * math.log(1.0 - r / float(max_reduction))
    return float(max(inv, 0.0))


def _auto_select_reduction_for_hub(
    model, X_encoded_base, df_meta, encoded_names, source_port, initial_shock,
    max_reduction=DEFAULT_MAX_REDUCTION, grid_points=GRID_POINTS, elbow_ratio=ELBOW_THRESHOLD_RATIO
) -> Tuple[float, pd.DataFrame]:
    """Simplified elbow-based auto-selection. Returns (selected_fraction, grid_df)."""
    try:
        Ip_baseline, _ = calculate_shock_intensity(model, X_encoded_base, df_meta, encoded_names, source_port, initial_shock)
        fractions = np.linspace(0.01, max_reduction, grid_points)
        ips = []
        for f in fractions:
            s = initial_shock * (1.0 - float(f))
            Ip_r, _ = calculate_shock_intensity(model, X_encoded_base, df_meta, encoded_names, source_port, s)
            ips.append(float(Ip_r))
        ips = np.array(ips)
        marginal_benefit = Ip_baseline - ips
        incremental = np.diff(marginal_benefit, prepend=0.0)
        # elbow detection
        initial_incremental = incremental[1] if incremental.size > 1 else (incremental[0] if incremental.size > 0 else 0.0)
        initial_incremental = max(initial_incremental, 1e-12)
        threshold = elbow_ratio * initial_incremental
        elbow_idx = next((i for i in range(1, len(incremental)) if incremental[i] <= threshold), None)
        if elbow_idx is None:
            # fallback: choose index achieving 80% of max benefit
            max_benefit = marginal_benefit.max() if marginal_benefit.size > 0 else 0.0
            target = 0.8 * max_benefit
            candidates = np.where(marginal_benefit >= target)[0]
            selected_idx = int(candidates[0]) if candidates.size > 0 else (len(fractions) - 1)
        else:
            selected_idx = elbow_idx
        selected_fraction = float(fractions[selected_idx])
        df_grid = pd.DataFrame({
            'reduction_fraction': fractions,
            'reduced_shock': initial_shock * (1.0 - fractions),
            'Ip_reduced': ips,
            'marginal_benefit': marginal_benefit,
            'incremental': incremental
        })
        return selected_fraction, df_grid
    except Exception:
        traceback.print_exc()
        return RESILIENCE_IMPROVEMENT_FACTOR, pd.DataFrame()


def plot_investment_ranking(df_investment: pd.DataFrame):
    """Horizontal bar chart."""
    if df_investment.empty:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    df_plot = df_investment.sort_values(by='Global_Risk_Reduction', ascending=False).reset_index(drop=True)
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("Greens", n_colors=len(df_plot))
    sns.barplot(data=df_plot.iloc[::-1].reset_index(drop=True), y='Source_Hub', x='Global_Risk_Reduction', palette=palette)
    for i, row in df_plot.iloc[::-1].reset_index(drop=True).iterrows():
        plt.text(row['Global_Risk_Reduction'] + 1e-6, i, f"{row['Global_Risk_Reduction']:.6f}", va='center')
    plt.title("Optimal Resilience Investment Portfolio: Marginal ROI Ranking")
    plt.xlabel("Global Systemic Risk Reduction (Total ΔI_p)")
    plt.ylabel("Source Hub")
    out = RESULTS_DIR / "resilience_investment_ranking.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"\n   (v) Optimal Investment Ranking Plot saved to: {out}")


def _compute_pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    data = df[['Investment_Amount', 'Global_Risk_Reduction']].to_numpy()
    dominated = np.zeros(len(data), dtype=bool)
    for i in range(len(data)):
        for j in range(len(data)):
            if i == j:
                continue
            inv_i, red_i = data[i]
            inv_j, red_j = data[j]
            if (inv_j <= inv_i and red_j >= red_i) and (inv_j < inv_i or red_j > red_i):
                dominated[i] = True
                break
    return df.loc[~dominated].copy()


# Main entry 
def execute_investment_optimization(
    df_panel: pd.DataFrame,
    feature_names_cleaned: List[str],
    preprocessor: Any,
    auto_select: Optional[bool] = True,
    total_budget: Optional[float] = None,
    investment_scale: float = DEFAULT_INVESTMENT_SCALE,
    max_reduction: float = DEFAULT_MAX_REDUCTION
) -> pd.DataFrame:
    """
    Run EVR optimization and return a DataFrame with columns:
    ['Investment_Rank','Source_Hub','Investment_Amount','Selected_Reduction_Fraction',
     'Initial_Ip','Global_Risk_Reduction','ROI_per_unit_investment']
    """
    print("\n" + "-" * 100)
    print("ECONOMIC VALUE OF RESILIENCE (EVR) OPTIMIZATION")
    print("-" * 100)

    # Load model and encoded matrix
    try:
        df_raw, df_meta, relevant_features = get_base_features_template(df_panel)
        expected_input_features = getattr(preprocessor, "input_feature_names_", relevant_features)
        df_align = df_raw[[c for c in expected_input_features if c in df_raw.columns]].copy()
        X_encoded_base = preprocessor.transform(df_align)
        encoded_names = list(preprocessor.get_feature_names_out())

        primary_model = joblib.load(PREDICTOR_MODEL_PATH)
        model_expected_n = getattr(primary_model, "n_features_in_", X_encoded_base.shape[1])
        X_encoded_base, encoded_names = _reconcile_encoded_matrix(X_encoded_base, encoded_names, model_expected_n, feature_names_cleaned)
    except Exception as e:
        print(f"   (x) Critical setup failed: {e}")
        traceback.print_exc()
        return pd.DataFrame()

    # Baseline cascades
    baseline_summary = []
    for hub in CRITICAL_HUBS_TO_TEST:
        Ip_initial, affected_count = calculate_shock_intensity(
            primary_model, X_encoded_base, df_meta, encoded_names, hub['source_port'], hub['shock'], quiet=True)
        baseline_summary.append({
            "Source_Hub": hub['source_port'],
            "Initial_Shock": hub['shock'],
            "Initial_Ip": Ip_initial,
            "Affected_Ports_Initial": affected_count
        })

    df_baseline = pd.DataFrame(baseline_summary)
    print("\n" + "-" * 100)
    print("BASELINE CASCADES (Initial State)")
    print("-" * 100)
    print("This shows the baseline impact before any resilience investment.")
    print(df_baseline[['Source_Hub', 'Initial_Shock', 'Affected_Ports_Initial']].to_string(index=False))
    print("\n" + "-" * 100)
    print("INVESTMENT SIMULATION (Marginal ROI)")
    print("-" * 100)

    # Generate recommended investments (internal)
    recommended = {}
    for hub in CRITICAL_HUBS_TO_TEST:
        source = hub['source_port']
        initial_shock = hub['shock']
        if auto_select:
            sel_frac, _ = _auto_select_reduction_for_hub(primary_model, X_encoded_base, df_meta, encoded_names, source, initial_shock, max_reduction, GRID_POINTS, ELBOW_THRESHOLD_RATIO)
        else:
            sel_frac = RESILIENCE_IMPROVEMENT_FACTOR
        required_inv = _reduction_to_investment(sel_frac, investment_scale, max_reduction)
        recommended[source] = {"selected_fraction": sel_frac, "required_investment": required_inv}

    # Build investment_map and optionally scale to total_budget
    raw_map = {hub: recommended[hub]['required_investment'] for hub in recommended}
    sum_raw = sum(raw_map.values())
    if total_budget is not None and sum_raw > 0:
        scale = float(total_budget) / float(sum_raw)
        investment_map = {k: float(v) * scale for k, v in raw_map.items()}
        print(f"\n   (i) Scaling recommended investments to total_budget={total_budget:.3f} (scale factor {scale:.4f})")
    else:
        investment_map = raw_map

    # Run per-hub simulation using investment_map
    results = []
    for i, hub in enumerate(CRITICAL_HUBS_TO_TEST):
        source = hub['source_port']
        initial_shock = hub['shock']
        Ip_initial = baseline_summary[i]['Initial_Ip']

        investment_amount = float(investment_map.get(source, 0.0))
        selected_frac = _investment_to_reduction(investment_amount, investment_scale, max_reduction)
        reduced_shock = initial_shock * (1.0 - selected_frac)

        print(f"\n   -> INVESTING in {source}: amount={investment_amount:.3f} -> predicted reduction={selected_frac*100:.2f}% -> new shock={reduced_shock:.3f}")

        Ip_reduced, affected_reduced = calculate_shock_intensity(primary_model, X_encoded_base, df_meta, encoded_names, source, reduced_shock, quiet=True)
        risk_reduction = float(Ip_initial - Ip_reduced)
        roi_per_unit = (risk_reduction / investment_amount) if investment_amount > 0 else np.nan

        # Escalate if no detectable benefit
        if risk_reduction == 0.0 and investment_amount >= 0.0:
            explore_fracs = np.linspace(max(0.001, selected_frac), max_reduction, max(5, GRID_POINTS))
            for ef in explore_fracs:
                test_inv = _reduction_to_investment(ef, investment_scale, max_reduction)
                if test_inv <= investment_amount + 1e-12:
                    continue
                test_shock = initial_shock * (1.0 - ef)
                test_Ip, _ = calculate_shock_intensity(primary_model, X_encoded_base, df_meta, encoded_names, source, test_shock, quiet=True)
                test_reduction = float(Ip_initial - test_Ip)
                if test_reduction > 0.0:
                    print(f"      (i) Escalated investment for {source} to {test_inv:.3f} to reach detectable EVR at reduction {ef*100:.2f}%")
                    investment_amount = float(test_inv)
                    selected_frac = float(ef)
                    reduced_shock = test_shock
                    Ip_reduced = float(test_Ip)
                    risk_reduction = float(test_reduction)
                    roi_per_unit = (risk_reduction / investment_amount) if investment_amount > 0 else np.nan
                    investment_map[source] = investment_amount
                    break

        print(f"      - Initial Ip: {Ip_initial:.6f} (Affected: {baseline_summary[i]['Affected_Ports_Initial']})")
        print(f"      - Reduced Ip: {Ip_reduced:.6f} (Affected: {affected_reduced})")
        print(f"      - Marginal Reduction (EVR): {risk_reduction:.6f}")
        if investment_amount > 0:
            print(f"      - Investment Amount: {investment_amount:.3f} -> ROI (ΔIp per unit): {roi_per_unit:.6f}")
        else:
            print(f"      - Investment Amount: 0.000 (no explicit investment provided)")

        results.append({
            "Source_Hub": source,
            "Investment_Amount": investment_amount,
            "Selected_Reduction_Fraction": selected_frac,
            "Initial_Ip": Ip_initial,
            "Reduced_Ip": Ip_reduced,
            "Global_Risk_Reduction": risk_reduction,
            "ROI_per_unit_investment": roi_per_unit
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='Global_Risk_Reduction', ascending=False).reset_index(drop=True)
    df_results['Investment_Rank'] = df_results.index + 1

    # Persist & pareto
    pareto_df = _compute_pareto_frontier(df_results)
    rec_out = RESULTS_DIR / "investment_recommendations.csv"
    df_results.to_csv(rec_out, index=False)
    pareto_out = RESULTS_DIR / "investment_pareto_frontier.csv"
    pareto_df.to_csv(pareto_out, index=False)
    print(f"\n   (v) Investment recommendations saved to: {rec_out}")
    print(f"   (v) Pareto frontier saved to: {pareto_out}")

    # Print formatted table 
    df_print = df_results.copy()
    df_print['Selected_Reduction_Fraction'] = df_print['Selected_Reduction_Fraction'].apply(lambda x: f"{x*100:.2f}%")
    df_print['Initial_Ip'] = df_print['Initial_Ip'].apply(lambda x: f"{x:.6f}")
    df_print['Global_Risk_Reduction'] = df_print['Global_Risk_Reduction'].apply(lambda x: f"{x:.6f}")
    df_print['ROI_per_unit_investment'] = df_print['ROI_per_unit_investment'].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "NA")

    print("\n" + "-" * 100)
    print("OPTIMAL RESILIENCE INVESTMENT PORTFOLIO (Marginal EVR & ROI)")
    print("-" * 100)
    display_cols = ['Investment_Rank', 'Source_Hub', 'Investment_Amount', 'Selected_Reduction_Fraction', 'Initial_Ip', 'Global_Risk_Reduction', 'ROI_per_unit_investment']
    print(df_print[display_cols].to_string(index=False))

    # Visualization 
    plot_investment_ranking(df_results)
    print("\n   (v) Investment Optimization Analysis Complete.")
    return df_results[['Investment_Rank', 'Source_Hub', 'Investment_Amount', 'Selected_Reduction_Fraction', 'Initial_Ip', 'Global_Risk_Reduction', 'ROI_per_unit_investment']]