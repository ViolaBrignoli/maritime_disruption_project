"""
MARITIME DISRUPTION PROJECT - Fragility Mapper (Quadrant Visualization)

This module computes structural fragility metrics for a configured set of
critical source hubs and produces a Strategic Quadrant Map (Network Resilience Fingerprint) that supports fast strategic diagnosis.

Key behaviours:
- Aligns and encodes the canonical input panel using the provided preprocessor,
  reconciles the encoded matrix to the deployed model's expected shape, and
  loads the persisted prediction model.
- For each configured critical source hub it:
    * Runs the cascade simulator (the "orchestrator" run_cascading_prediction)
      to obtain per-port risk deltas produced by a shock at that hub.
    * Computes three succinct hub-level metrics:
        - Prediction Intensity (Ip): total network risk increase produced (Σ ΔP)
        - Prediction Distance (Dp): number of ports with measurable increase
          above a negligible threshold (structural reach)
        - Structural Fragility Index (I_fragility): Ip / (Dp + ε) — a
          concentration metric that highlights high-impact, concentrated failures.
- Produces a Strategic Quadrant Map:
    - X-axis: Prediction Distance (Dp)
    - Y-axis: Prediction Intensity (Ip)
    - Bubble size: Fragility_Index
    - Bubble color: Source hub identity
    - Median lines split the plot into four quadrants and soft background shading
      emphasizes strategic zones (e.g., "Global Contagion", "Fragility Bottleneck").

Returns: 
Writes a high-resolution PNG quadrant map to results/structural_fragility_map.png.

Primary inputs:
- df_panel
- preprocessor
- primary model artifact
- all_stress_test_results

The "Orchestrator":
- The orchestrator is the cascade runner located in src.cascading_predictor
- Fragility mapper calls run_cascading_prediction internally for each source
  hub to compute the per-hub per-port deltas used to compute Ip, Dp, and the
  fragility index.

Key computations and interpretation:
- Intensity (Ip): sum of Risk_Increase across affected ports. Higher Ip means
  larger absolute network impact; Ip to prioritize hubs that create the
  most total expected harm.
- Distance (Dp): count of ports with Risk_Increase >= NEGLIGIBLE_RISK_THRESHOLD.
  A larger Dp indicates wider structural spread — useful for gauging scope of
  monitoring and containment.
- Fragility Index (I_fragility): Ip / (Dp + ε) — a concentration metric. Large
  values indicate hubs that produce high total impact concentrated on relatively
  few targets (fragility bottlenecks) and therefore candidates for targeted
  hardening.
- Quadrant Map interpretation:
    * Upper-right (high Dp, high Ip): Global contagion — high priority for
      mitigation and international coordination.
    * Upper-left (low Dp, high Ip): Fragility bottleneck — candidates for
      infrastructure hardening or capacity redundancy.
    * Lower quadrants: Informative for monitoring and tactical planning.
"""
from typing import Any, List, Dict, Tuple
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from src.cascading_predictor import run_cascading_prediction, PREDICTOR_MODEL_PATH
from src.cascading_predictor import get_base_features_template, _reconcile_encoded_matrix

# Configuration
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Experiment Configuration
CRITICAL_HUBS_TO_TEST = [
    {"source_port": "Singapore", "shock": 5.0},
    {"source_port": "Shanghai", "shock": 8.0},
    {"source_port": "Rotterdam", "shock": 5.0},
    {"source_port": "Suez", "shock": 6.0},
    {"source_port": "Durban", "shock": 7.0},
    {"source_port": "Valparaiso", "shock": 6.0},
]
# Threshold below which cascade influence is considered negligible 
NEGLIGIBLE_RISK_THRESHOLD = 0.001


def calculate_fragility_metrics(
    model: Any,
    X_encoded_base: np.ndarray,
    df_meta: pd.DataFrame,
    encoded_names: List[str],
    source_port: str,
    shock_magnitude: float
) -> Dict:
    """
    Runs one cascade and calculates Prediction Distance (Dp) and Intensity (Ip).
    Returns a dict with summary metrics.
    """
    try:
        all_results = run_cascading_prediction(
            model=model,
            X_encoded=X_encoded_base,
            df_meta=df_meta,
            disrupted_port_name=source_port,
            encoded_feature_names=encoded_names,
            shock_magnitude=shock_magnitude
        )
    except Exception:
        return {}

    if not all_results:
        return {}

    df_results = pd.DataFrame(all_results)
    df_results['Risk_Increase_Numeric'] = pd.to_numeric(
        df_results.get('Risk_Increase', 0),
        errors='coerce'
    ).fillna(0.0)

    # 1. Prediction Intensity (Ip): Total sum of all measured risk increases in the network
    Ip = df_results['Risk_Increase_Numeric'].sum()

    # 2. Prediction Distance (Dp): Count of ports where the influence is >= threshold
    df_significant = df_results[df_results['Risk_Increase_Numeric'] >= NEGLIGIBLE_RISK_THRESHOLD]
    Dp = len(df_significant)

    # 3. Structural Fragility Index (I_fragility)
    I_fragility = Ip / (Dp + 1e-6)

    return {
        "Source_Hub": source_port,
        "Intensity (Ip)": Ip,
        "Distance (Dp)": Dp,
        "Fragility_Index": I_fragility,
        "Shock_Magnitude": shock_magnitude,
        "Affected_Ports_Count": len(df_results)
    }


def plot_fragility_map(df_fragility: pd.DataFrame) -> None:
    """
    Creates the Network Resilience Quadrant Map (Area Chart) for strategic diagnosis.
    Legend (hub color) is placed outside the plot to avoid cropping.
    """
    if df_fragility is None or df_fragility.empty:
        print("   (i) Cannot generate Fragility Map: DataFrame is empty.")
        return

    # Styling
    sns.set_style("whitegrid")
    num_hubs = df_fragility['Source_Hub'].nunique()

    # Figure size tuned for clarity and to leave space for an outside legend
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate medians for quadrant split
    median_ip = float(df_fragility['Intensity (Ip)'].median())
    median_dp = float(df_fragility['Distance (Dp)'].median())

    # 1. Draw Strategic Zones (Background Colors for Quadrant Analysis) 
    max_dp = (
        float(df_fragility['Distance (Dp)'].max()) * 1.15
        if df_fragility['Distance (Dp)'].max() > 0
        else median_dp + 1.0
    )
    max_ip = (
        float(df_fragility['Intensity (Ip)'].max()) * 1.15
        if df_fragility['Intensity (Ip)'].max() > 0
        else median_ip + 1.0
    )

    ax.set_xlim(0, max_dp)
    ax.set_ylim(0, max_ip)

    # Quadrant colors and labels
    Q_SETTINGS = {
        "GLOBAL CONTAGION": (median_dp, max_dp, median_ip, max_ip, '#DC143C', 'Strategy I: Mitigation'),  # Crimson Red
        "FRAGILITY BOTTLENECK": (0, median_dp, median_ip, max_ip, '#FF8C00', 'Strategy II: Hardening'),  # Dark Orange
    }

    for title, (xmin, xmax, ymin, ymax, color, subtitle) in Q_SETTINGS.items():
        ax.axvspan(xmin, xmax, ymin=ymin / max_ip, ymax=ymax / max_ip, color=color, alpha=0.12, zorder=1)
        x_pos = xmin + (xmax - xmin) * 0.5
        y_pos = ymax * 0.95
        ax.text(x_pos, y_pos, title.upper(), color='black', fontsize=11, weight='bold', ha='center', va='top', zorder=2)
        ax.text(x_pos, y_pos * 0.9, subtitle, color='black', fontsize=9, ha='center', va='top', zorder=2)

    # 2. Plot Data Points (Bubbles)
    # Create a palette with num_hubs colors
    palette = sns.color_palette('tab10', n_colors=max(10, num_hubs))
    hub_order = list(df_fragility['Source_Hub'].unique())
    hub_color_map = {hub: palette[i % len(palette)] for i, hub in enumerate(hub_order)}

    # Use seaborn to plot points but suppress the automatic legend
    sns.scatterplot(
        data=df_fragility,
        x='Distance (Dp)',
        y='Intensity (Ip)',
        size='Fragility_Index',
        sizes=(200, 2200),
        hue='Source_Hub',
        palette=hub_color_map,
        alpha=0.9,
        legend=False,  # we'll create a custom legend below (hub names will be in legend only)
        ax=ax,
        edgecolor='black',
        linewidth=0.9,
        zorder=3
    )


    # 3. Add Strategic Quadrant Lines (The Split)
    ax.axvline(median_dp, color='k', linestyle='--', linewidth=1.3, alpha=0.85, zorder=2)
    ax.axhline(median_ip, color='k', linestyle='--', linewidth=1.3, alpha=0.85, zorder=2)

    # 4. Final Touches and Legend Creation 
    ax.set_title(
        "Network Resilience Fingerprint: Structural Fragility Mapping",
        fontsize=14,
        fontweight='bold',
        pad=12
    )
    ax.set_xlabel(
        f"Prediction Distance (Dp) - Affected Ports (P ≥ {NEGLIGIBLE_RISK_THRESHOLD:.3f})",
        fontsize=11,
        fontweight='bold'
    )
    ax.set_ylabel(
        "Prediction Intensity (Ip) - Total Network Risk Caused (Σ ΔP)",
        fontsize=11,
        fontweight='bold'
    )

    ax.grid(True, linestyle=':', alpha=0.5, zorder=0)

    # Build legend entries for hubs (color only)
    legend_handles = []
    for hub in hub_order:
        color = hub_color_map[hub]
        patch = Patch(facecolor=color, edgecolor='black', label=hub)
        legend_handles.append(patch)

    # Place legend below the axes (figure-level) and adapt columns to the number of hubs
    ncol = min(6, max(1, len(legend_handles)))
    fig.legend(
        handles=legend_handles,
        title="Source Hub (color)",
        loc='lower center',
        bbox_to_anchor=(0.5, -0.08),
        ncol=ncol,
        fontsize=10,
        title_fontsize=11,
        frameon=True
    )

    # Ensure there's enough bottom margin for the legend and nothing gets cut off
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)

    # Save 
    file_path = RESULTS_DIR / "structural_fragility_map.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def execute_fragility_analysis(
    df_panel: pd.DataFrame,
    feature_names_cleaned: List[str],
    preprocessor: Any,
    all_stress_test_results: List[Dict]
) -> pd.DataFrame:
    """
    Orchestrates the Fragility Mapping experiment and generates the quadrant map.
    Returns df_fragility (per-source summary). 
    """
    print("\n" + "-" * 100)
    print("STRUCTURAL RESILIENCE: FRAGILITY MAPPING (QUADRANT)")
    print("-" * 100)

    # 1. Load Data and Reconcile Matrix
    try:
        df_raw, df_meta, relevant_features = get_base_features_template(df_panel)

        expected_input_features = getattr(preprocessor, "input_feature_names_", relevant_features)
        df_align = df_raw[[c for c in expected_input_features if c in df_raw.columns]].copy()
        X_encoded_base = preprocessor.transform(df_align)
        encoded_names = list(preprocessor.get_feature_names_out())

        primary_model = joblib.load(PREDICTOR_MODEL_PATH)
        model_expected_n = getattr(primary_model, "n_features_in_", X_encoded_base.shape[1])
        X_encoded_base, encoded_names = _reconcile_encoded_matrix(
            X_encoded_base, encoded_names, model_expected_n, feature_names_cleaned
        )

    except Exception as e:
        print(f"   (x) Critical setup failed: {e}")
        traceback.print_exc()
        return pd.DataFrame()

    analysis_results = []

    # 2. Run Fragility Calculation for each Hub
    for hub in CRITICAL_HUBS_TO_TEST:
        metrics = calculate_fragility_metrics(
            model=primary_model,
            X_encoded_base=X_encoded_base,
            df_meta=df_meta,
            encoded_names=encoded_names,
            source_port=hub['source_port'],
            shock_magnitude=hub['shock']
        )

        if metrics:
            analysis_results.append(metrics)
        else:
            print(f"      (x) Calculation failed for {hub['source_port']}.")

    # 3. Final Fragility Report (Table)
    df_fragility = pd.DataFrame(analysis_results)

    if df_fragility.empty:
        print("   (i) No results generated for Fragility Analysis.")
        return pd.DataFrame()

    df_fragility = df_fragility.sort_values(by='Fragility_Index', ascending=False).reset_index(drop=True)
    df_fragility['Fragility_Rank'] = df_fragility.index + 1

    # 4. Execute Visualization (quadrant only)
    plot_fragility_map(df_fragility)

    print(f"\n(v) Structural Fragility Map saved to: {RESULTS_DIR / 'structural_fragility_map.png'}")
    return df_fragility