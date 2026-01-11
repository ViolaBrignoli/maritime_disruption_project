"""
MARITIME DISRUPTION PROJECT - Evaluation Module

This module evaluates trained binary classifiers for the maritime disruption
project, producing numeric diagnostics, visual artifacts, and a concise
summary table that can be used to choose and persist the best candidate
model for downstream simulation and decision-making.

Purpose:
    - Compute a broad set of performance and calibration metrics that are
      meaningful for rare-event / risk prediction (ROC AUC, PR AUC, Gini,
      Brier score, Expected Calibration Error (ECE), top-decile lift).
    - Find an operating probability threshold tuned to a user-specified
      F-beta (default BETA = 0.5) and produce classification reports at that
      threshold for operational evaluation.
    - Produce and save diagnostic plots (confusion matrix heatmap, precision-
      recall curve, threshold tuning plot, reliability / calibration diagram,
      lift/capture chart, and risk-quantile enrichment barplot) to the results/
      directory for inspection and reporting.
    - Persist model artifacts: the module saves a standalone XGBoost artifact 
      for special downstream analysis and also saves the best
      model (by ROC AUC) and a small metadata JSON describing the selection.
    - Provide robust fallbacks for models that do not expose predict_proba:
      when only discrete predictions or decision_function scores are available,
      it converts them to a pragmatic probability proxy so metrics and plots
      can still be generated.

Primary public function:
    generate_comprehensive_report(models_dict, X_test, y_test, feature_names=None)
    - Inputs:
        * models_dict: mapping from string model name -> trained estimator
        * X_test: feature matrix for evaluation (numpy array or pandas DataFrame)
        * y_test: true binary labels aligned with X_test
        * feature_names: optional list of names used for diagnostics/annotations
    - Outputs / side-effects:
        * Returns a pandas DataFrame summarizing per-model metrics (sorted by
          ROC AUC).
        * Writes artifacts into the local results/ directory:
            - final_model_comparison.csv (summary table)
            - best_model_final.joblib (pickled best model)
            - best_model_metadata.json (selection metadata)
            - xgb_model_final.joblib (if an XGBoost model was evaluated)
            - PNG files for each model's plots described above
        * Prints a readable console summary for each evaluated model.

Key computed metrics and utilities:
    - ROC AUC, PR AUC (average precision), Gini coefficient
    - Brier score (mean squared error of probabilities)
    - Expected Calibration Error (ECE) using binning
    - Top-decile lift (concentration of positives in top 10% probabilities)
    - Optimal probability threshold by maximizing F-beta (configurable BETA)
    - Classification report at the chosen threshold (precision/recall/F1)
    - Several plotting utilities to visualize performance and calibration

Robustness & behavior notes:
    - The module clips predicted probabilities to [0, 1] and falls back to
      decision_function outputs (min-max scaled) or discrete predictions when
      probabilistic outputs are not available.
    - The threshold search and F-beta computation guard against degenerate
      cases (no thresholds, constant scores) and avoid division-by-zero.
    - Plotting functions are tolerant of empty bins and missing data; failures
      in individual plots are caught and logged without aborting the full
      evaluation.
    - All artifacts are saved under the results/ directory; the module creates
      this directory if it does not already exist.
"""
from pathlib import Path
from typing import Any, List, Optional, Dict
import traceback
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
    roc_auc_score, precision_recall_curve, average_precision_score,
    brier_score_loss, precision_score, recall_score
)
from sklearn.calibration import calibration_curve

# Configuration and Path
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set plotting style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.2)

#  F-BETA CONFIGURATION 
BETA = 0.5 # Prioritizes Precision (BETA < 1)

def _to_numpy(X):
    if isinstance(X, pd.DataFrame):
        return X.values.astype(float)
    if isinstance(X, np.ndarray):
        return X.astype(float)
    return np.asarray(X).astype(float)


def get_model_probabilities(model: Any, X_test: Any) -> Optional[np.ndarray]:
    """
    Return predicted probabilities for the positive class (1).
    """
    X_np = _to_numpy(X_test)
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_np)[:, 1]
            return np.clip(probs, 0.0, 1.0)
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_np)
            # Min-max scale to [0,1] as a pragmatic fallback
            if np.all(scores == scores[0]):
                return np.full_like(scores, 0.5, dtype=float)
            scaled = (scores - scores.min()) / (scores.max() - scores.min())
            return np.clip(scaled, 0.0, 1.0)
    except Exception:
        pass
    return None


def calculate_f_beta_score(precision: np.ndarray, recall: np.ndarray, beta: float) -> np.ndarray:
    """Calculates the F-beta score array."""
    # F-beta = (1 + beta^2) * P * R / (beta^2 * P + R)
    beta2 = beta ** 2
    denominator = (beta2 * precision + recall + 1e-12)
    return (1.0 + beta2) * (precision * recall) / denominator


def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray, beta: float) -> (float, float):
    """
    Return the probability threshold that maximizes F-beta score.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    # Calculate F-beta scores
    f_beta_scores = calculate_f_beta_score(precision, recall, beta)
    # align thresholds (f_beta_scores length is len(thresholds)+1)
    if thresholds.size == 0:
        thresh = 0.5
        preds = (y_probs >= thresh).astype(int)
        # Recalculate F-beta for the single threshold
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        opt_f_beta = calculate_f_beta_score(np.array([p]), np.array([r]), beta)[0]
        return thresh, float(opt_f_beta)
    f_beta_trim = f_beta_scores[:-1]
    idx = int(np.nanargmax(f_beta_trim))
    return float(thresholds[idx]), float(f_beta_trim[idx])

# Plotting helpers (save to results/)
def _savefig(fname: str):
    full = RESULTS_DIR / fname
    plt.tight_layout()
    plt.savefig(full, dpi=300)
    plt.close()


def plot_confusion_matrix_heatmap(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, suffix: str = "optimal"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Normal (0)", "Disrupted (1)"],
                yticklabels=["Normal (0)", "Disrupted (1)"])
    plt.title(f"Confusion Matrix: {model_name} ({suffix})", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    _savefig(f"confusion_matrix_{model_name}_{suffix}.png")


def plot_precision_recall_curve(y_true: np.ndarray, y_probs: np.ndarray, model_name: str):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, lw=2, color="darkorange", label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall: {model_name}", fontsize=13)
    plt.legend()
    _savefig(f"pr_curve_{model_name}.png")


def plot_threshold_tuning(y_true: np.ndarray, y_probs: np.ndarray, model_name: str, beta: float):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    if thresholds.size == 0:
        return
    f_beta_scores = calculate_f_beta_score(precision, recall, beta)
    f_beta_trim = f_beta_scores[:-1]
    # Handle NaN/Inf due to zero division carefully
    f_beta_trim = np.nan_to_num(f_beta_trim, nan=0.0, posinf=0.0, neginf=0.0)
    if not f_beta_trim.any():
        return # Plotting is meaningless if all scores are zero
    best_idx = int(np.nanargmax(f_beta_trim))
    best_thresh = float(thresholds[best_idx])
    best_f_beta = float(f_beta_trim[best_idx])

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f_beta_trim, color="teal", lw=2)
    plt.axvline(best_thresh, color="red", linestyle="--", label=f"Best p = {best_thresh:.4f}")
    plt.scatter([best_thresh], [best_f_beta], color="red", zorder=5)
    plt.xlabel("Probability Threshold")
    plt.ylabel(f"F{beta} Score (Prioritizes Precision)")
    plt.title(f"Threshold Tuning (F{beta}) - {model_name}")
    plt.legend()
    _savefig(f"threshold_tuning_{model_name}_f{beta}.png")


def plot_calibration_curve(y_true: np.ndarray, y_probs: np.ndarray, model_name: str):
    # reliability diagram
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
    plt.figure(figsize=(7, 7))
    plt.plot(prob_pred, prob_true, "o-", label=model_name)
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Calibration Curve - {model_name}")
    plt.legend()
    _savefig(f"calibration_curve_{model_name}.png")


def plot_risk_quantile_enrichment(y_true: np.ndarray, y_probs: np.ndarray, model_name: str):
    """
    Bar plot showing observed disruption rates in high-risk quantiles (Top 1%, 5%, 10%)
    vs base rate. This communicates how concentrated risk predictions are.
    """
    df = pd.DataFrame({"true": y_true, "prob": y_probs})
    base_rate = df["true"].mean() * 100.0
    quantiles = [0.99, 0.95, 0.90]
    labels = ["Top 1%", "Top 5%", "Top 10%"]
    rates = []
    for q in quantiles:
        thr = df["prob"].quantile(q)
        sel = df[df["prob"] >= thr]
        rates.append(float(sel["true"].mean() * 100.0) if not sel.empty else 0.0)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=labels, y=rates, palette="Reds")
    plt.axhline(base_rate, color="navy", linestyle="--", label=f"Base rate {base_rate:.2f}%")
    plt.ylabel("Observed Disruption Rate (%)")
    plt.title(f"Observed Disruption Rate in Top Risk Quantiles - {model_name}")
    plt.legend()
    _savefig(f"risk_quantile_enrichment_{model_name}.png")


def plot_lift_chart(y_true: np.ndarray, y_probs: np.ndarray, model_name: str, n_bins: int = 10):
    """
    Lift chart: compare cumulative capture of positives by deciles.
    """
    df = pd.DataFrame({"true": y_true, "prob": y_probs}).sort_values("prob", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index + 1, q=n_bins, labels=False) # top bucket = decile 0
    grouped = df.groupby("decile")["true"].agg(["sum", "count"]).reset_index().sort_values("decile")
    grouped["capture_rate"] = grouped["sum"].cumsum() / df["true"].sum()
    grouped["bucket_pct"] = grouped["count"] / df.shape[0]
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, n_bins + 1), grouped["capture_rate"], marker="o", label="Model capture")
    plt.plot(range(1, n_bins + 1), np.linspace(0, 1, n_bins), linestyle="--", color="gray", label="Random")
    plt.xlabel("Bucket (1 = highest risk)")
    plt.ylabel("Cumulative Capture of True Positives")
    plt.title(f"Lift / Capture Chart - {model_name}")
    plt.legend()
    _savefig(f"lift_chart_{model_name}.png")

# Additional numeric diagnostics
def compute_brier(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_probs))


def compute_ece(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE): weighted average of |conf - acc| across bins.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_probs >= lo) & (y_probs < hi) if i < n_bins - 1 else (y_probs >= lo) & (y_probs <= hi)
        if not mask.any():
            continue
        prop = mask.sum() / n
        acc = y_true[mask].mean()
        conf = y_probs[mask].mean()
        ece += prop * abs(acc - conf)
    return float(ece)


def compute_top_decile_lift(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """
    Top-decile lift: ratio of positive rate in top 10% vs overall positive rate.
    """
    df = pd.DataFrame({"true": y_true, "prob": y_probs})
    thr = df["prob"].quantile(0.90)
    top = df[df["prob"] >= thr]
    if top.empty:
        return 0.0
    top_rate = top["true"].mean()
    base_rate = df["true"].mean()
    return float(top_rate / (base_rate + 1e-12))


# Main reporting function
def generate_comprehensive_report(models_dict: Dict[str, Any], X_test: Any, y_test: Any, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Evaluate models, produce professional artifacts, save a concise summary CSV,
    and persist the best model (by ROC AUC) as results/best_model_final.joblib along
    with a small metadata JSON for downstream use.
    """
    try:

        X_test_np = _to_numpy(X_test)
        y_test_arr = np.asarray(y_test).astype(int)

        summary_rows = []

        for model_name, model in models_dict.items():
            print(f"\n>> Evaluating {model_name.upper()} ...")

            y_probs = get_model_probabilities(model, X_test_np)
            if y_probs is None:
                # fallback to discrete predictions only
                y_pred = model.predict(X_test_np)
                y_probs = np.where(y_pred == 1, 1.0, 0.0)

            # Core metrics
            roc = float(roc_auc_score(y_test_arr, y_probs))
            pr_auc = float(average_precision_score(y_test_arr, y_probs))
            gini = 2.0 * roc - 1.0
            brier = compute_brier(y_test_arr, y_probs)
            ece = compute_ece(y_test_arr, y_probs, n_bins=10)
            top_decile_lift = compute_top_decile_lift(y_test_arr, y_probs)

            # Optimal threshold and classification report (now based on F-BETA)
            opt_thresh, opt_f_beta = find_optimal_threshold(y_test_arr, y_probs, BETA)
            y_pred_opt = (y_probs >= opt_thresh).astype(int)
            acc_opt = float(accuracy_score(y_test_arr, y_pred_opt))
            # Recalculate F1 for display purposes
            opt_f1 = float(f1_score(y_test_arr, y_pred_opt, zero_division=0))
            # Print summary to console
            print(f" ROC AUC: {roc:.4f} | PR AUC: {pr_auc:.4f} | Gini: {gini:.4f}")
            print(f" Brier Score: {brier:.5f} | ECE: {ece:.5f} | Top-decile Lift: {top_decile_lift:.2f}")
            print(f" Optimal F{BETA}: {opt_f_beta:.4f} at threshold {opt_thresh:.4f}")
            # Report F1 as traditional metric for comparison
            print(f" Corresponding F1: {opt_f1:.4f}")
            print("\nClassification Report (at F{BETA} optimal threshold):")
            print(classification_report(y_test_arr, y_pred_opt, zero_division=0))

            # Add to summary rows
            summary_rows.append({
                "Model": model_name,
                "ROC_AUC": roc,
                "PR_AUC": pr_auc,
                "GINI": gini,
                "Brier": brier,
                "ECE": ece,
                "TopDecileLift": top_decile_lift,
                "OptimalThreshold_F0.5": opt_thresh,
                "F0.5_Optimal": opt_f_beta,
                "F1_Corresponding": opt_f1,
                "Accuracy_Optimal": acc_opt
            })
            
            # Save the XGBoost model specifically for Dual-Model Analysis 
            if model_name.lower() == 'xgb':
                xgb_path = RESULTS_DIR / "xgb_model_final.joblib"
                try:
                    joblib.dump(model, xgb_path)
                    print(f" (v) XGBoost model saved for Dual-Model Analysis to: {xgb_path}")
                except Exception as e_xgb_save:
                    print(f" (warn) Could not save XGBoost artifact for Dual-Model Analysis: {e_xgb_save}")

            
            # Generate artifacts (plots)
            try:
                plot_confusion_matrix_heatmap(y_test_arr, y_pred_opt, model_name, suffix=f"p{opt_thresh:.4f}_F{BETA}")
                plot_precision_recall_curve(y_test_arr, y_probs, model_name)
                plot_threshold_tuning(y_test_arr, y_probs, model_name, beta=BETA) # Plot F0.5
                plot_calibration_curve(y_test_arr, y_probs, model_name)
                plot_lift_chart(y_test_arr, y_probs, model_name, n_bins=10)
                plot_risk_quantile_enrichment(y_test_arr, y_probs, model_name)
            except Exception as e_plot:
                print(f" [Warning] Plotting failed for {model_name}: {e_plot}")

        # Save summary dataframe
        df_summary = pd.DataFrame(summary_rows).sort_values(by="ROC_AUC", ascending=False).reset_index(drop=True)
        summary_path = RESULTS_DIR / "final_model_comparison.csv"
        df_summary.to_csv(summary_path, index=False)

        # Persist the best model (by ROC_AUC) and metadata
        if not df_summary.empty:
            best_row = df_summary.iloc[0]
            best_model_name = best_row["Model"]
            best_model = models_dict.get(best_model_name)
            if best_model is not None:
                model_path = RESULTS_DIR / "best_model_final.joblib"
                try:
                    joblib.dump(best_model, model_path)
                    # Save small metadata JSON
                    metadata = {
                        "best_model_name": best_model_name,
                        "selection_metric": "ROC_AUC",
                        "roc_auc": float(best_row["ROC_AUC"]),
                        "pr_auc": float(best_row["PR_AUC"]),
                        "optimal_threshold_f05": float(best_row["OptimalThreshold_F0.5"]),
                        "f05_optimal": float(best_row["F0.5_Optimal"])
                    }
                    meta_path = RESULTS_DIR / "best_model_metadata.json"
                    with open(meta_path, "w") as fh:
                        json.dump(metadata, fh, indent=2)
                    print(f"\n (v) Best model saved to: {model_path}")
                    print(f" (v) Best model metadata saved to: {meta_path}")
                except Exception as e_save:
                    print(f" (warn) Could not save best model artifact: {e_save}")

        # Final formatted status
        print(f"Evaluation complete. Summary saved to: {summary_path}")
        print("Artifacts saved to the results/ directory (PNG files).")

        return df_summary

    except Exception as e:
        print(f"[ERROR] Evaluation pipeline failed: {e}")
        traceback.print_exc()
        raise