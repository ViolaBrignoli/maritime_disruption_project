"""
MARITIME DISRUPTION PROJECT - Stress Test Visualization (Cumulative Risk Plot)

Create a cumulative capture (gain) chart that shows how much of the total
cascade-induced risk (sum of risk increases across affected ports) would be
captured by investigating ports ranked by predicted risk increase.

**This script uses the structured list of enriched results (df_report) generated
by the main Orchestrator script cascading_predictor.py (which runs the cascade simulation and prediction)
as its primary input.**

Key behaviour:
    - This function plots all affected ports supplied in df_report.
      The X-axis is scaled so 100% equals the full set of affected ports passed
      into the function. 
    - Sorting uses Risk_Increase as the primary key and XGB_Risk_Prob as a
      secondary tie-breaker.
    - The chart communicates the concentration of risk across the entire returned set.

What the plot shows:
    - X-axis: Percentage of plotted ports investigated (rank / total_plotted * 100).
      plotted_count = number of affected ports passed in.
    - Y-axis: Cumulative percentage of the total predicted cascade risk captured
      by investigating the ports (cumulative sum of Risk_Increase divided by the
      total Risk_Increase across the full set).
    - A dashed vertical/horizontal marker shows where 50% of the total plotted
      risk is captured, with an annotation of how many ports are required to
      reach that coverage (if that threshold is reached).

Outputs:
    - Saves PNG to /results/multi_scenario_risk_summary.png
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Any

# Paths
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_stress_test_report(df_report: pd.DataFrame, file_name: str = "multi_scenario_risk_summary.png"):
    """
    Create and save a cumulative capture (gain) chart using all affected ports
    present in df_report. 
    """
    if df_report is None or df_report.empty:
        print("   (i) Cannot generate plot: Final report dataframe is empty.")
        return

    df = df_report.copy()

    # Convert to numeric 
    df["Risk_Increase_Numeric"] = pd.to_numeric(df.get("Risk_Increase"), errors="coerce").fillna(0.0)
    df["XGB_Risk_Prob_Numeric"] = pd.to_numeric(df.get("XGB_Risk_Prob"), errors="coerce").fillna(0.0)

    # Sort by Risk_Increase (primary) then XGB_Risk_Prob (secondary)
    df_sorted = df.sort_values(
        by=["Risk_Increase_Numeric", "XGB_Risk_Prob_Numeric"],
        ascending=False
    ).reset_index(drop=True)

    total_ports = len(df_sorted)
    print(f"   (i) Plotting all {total_ports} affected ports.")

    # Compute total risk and bail out if zero
    total_risk = df_sorted["Risk_Increase_Numeric"].sum()
    if total_risk == 0 or np.isclose(total_risk, 0.0):
        print("   (i) Total measured risk increase is zero. Skipping Cumulative Risk Plot.")
        return

    # Cumulative calculations
    df_sorted["Cumulative_Risk"] = df_sorted["Risk_Increase_Numeric"].cumsum()
    df_sorted["% Captured"] = (df_sorted["Cumulative_Risk"] / total_risk) * 100
    df_sorted["% Ports Investigated"] = (df_sorted.index + 1) / total_ports * 100

    # Plot style
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(12, 7))

    plt.plot(
        df_sorted["% Ports Investigated"],
        df_sorted["% Captured"],
        marker="o",
        markersize=3,
        lw=2,
        color="darkblue",
        label="Model Efficiency"
    )

    # Annotate 50% capture point (if present)
    try:
        idx_50 = df_sorted[df_sorted["% Captured"] >= 50].index[0]
        ports_50_pct_investigated = df_sorted.loc[idx_50, "% Ports Investigated"]
        ports_50_count = idx_50 + 1

        plt.axhline(50, color="gray", linestyle="--", lw=1)
        plt.axvline(ports_50_pct_investigated, color="gray", linestyle="--", lw=1,
                    label=f"50% Risk in {ports_50_count} Ports ({ports_50_pct_investigated:.1f}%)")
        plt.scatter(ports_50_pct_investigated, 50, color="red", s=50, zorder=5)
    except Exception:
        # If the 50% point is not reached -> silently continue
        pass

    # Titles and labels
    plt.title(f"Cumulative Cascade Risk Capture (All {total_ports} Affected Ports)", fontsize=14, fontweight="bold")
    plt.xlabel("Percentage of Affected Ports Investigated", fontsize=12)
    plt.ylabel("Cumulative Percentage of Cascade Risk Captured", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(loc="lower right")
    plt.xlim(0, 100)
    plt.ylim(0, 100)

    # Save
    file_path = RESULTS_DIR / file_name
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()

    print(f"\n   (v) Visualization (Cumulative Risk Plot) saved to: {file_path}")