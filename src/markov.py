"""
MARITIME DISRUPTION PROJECT - Markov Baseline Model 

Provide a compact, well-tested Markov-chain baseline for port-year panel data. 
This implementation is panel-aware (counts only within-port consecutive-year transitions), applies Laplace smoothing for numerical stability, and supports both "operational" and "pure forecast" prediction modes.

Key behaviours:
- Train transition probabilities P(next_state | current_state) from panel data, counting only consecutive-year transitions within the same port (join_key).
- Apply Laplace (additive) smoothing to avoid zero-counts and improve robustness.
- Predict per-port using each port's last observed training-state as the initial previous-state. 
  Two prediction modes are supported:
  * use_observed_prev=True  : assumes the true current state is observed at prediction time and is used to seed the next step.
  * use_observed_prev=False : roll-forward using model's own previous predictions.
- Provide wrapper utilities to accept either panel DataFrames (preferred) or simple 1D arrays for quick experiments.
- Return evaluation metrics plus helpful diagnostics (confusion matrix, transition matrix).

Note:
- The baseline is intentionally simple and interpretable but limited: it only uses the previous-year binary state to forecast the next-year binary state.
- Panel Data Version in Main Entry Point Execution.
"""

from typing import Dict, Tuple, Any, Iterable, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

EPS = 1e-12  # numerical guard

# Internal helpers
def _validate_binary_series(s: Iterable) -> np.ndarray:
    """
    Coerce an iterable of values to a binary numpy array of 0/1 ints.
    Non-zero values map to 1; NaNs and zeros map to 0.
    """
    arr = np.asarray(list(s))
    # Replace NaN-like with 0, then cast to int (truthy -> 1, else 0)
    arr = np.where(pd.isna(arr), 0, arr)
    try:
        arr = arr.astype(int)
    except Exception:
        # fall back to boolean casting for mixed types
        arr = np.where(arr != 0, 1, 0).astype(int)
    arr = np.where(arr != 0, 1, 0)  # ensure strictly 0/1
    return arr

# Training (panel-aware)
def train_markov_chain_panel(
    df: pd.DataFrame,
    group_col: str = "join_key",
    time_col: str = "year",
    target_col: str = "target_is_disrupted_in_year",
    laplace: float = 1.0
) -> np.ndarray:
    """
    Train a 2x2 Markov transition matrix P(next_state | current_state) from panel data.

    Counts transitions only for consecutive rows within each group (sorted by time_col).
    Applies additive Laplace smoothing before normalizing rows.

    Args:
        df: Panel DataFrame containing group_col, time_col, and target_col.
        group_col: Column name that identifies the panel entity (e.g., 'join_key').
        time_col: Column name representing time ordering (e.g., 'year').
        target_col: Binary target column (0/1).
        laplace: Additive smoothing constant (default 1.0).

    Returns:
        probs: 2x2 numpy array where row i is P(next=0/1 | current=i).
    """
    counts = np.zeros((2, 2), dtype=float)

    # Group by entity and count only consecutive-year transitions
    for _, grp in df.groupby(group_col):
        grp_sorted = grp.sort_values(time_col)
        y = _validate_binary_series(grp_sorted[target_col].values)
        if y.size < 2:
            continue
        curr = y[:-1]
        nxt = y[1:]
        for c, n in zip(curr, nxt):
            counts[c, n] += 1.0

    # Laplace smoothing and row normalization
    counts += laplace
    row_sums = counts.sum(axis=1, keepdims=True) + EPS
    probs = counts / row_sums
    return probs

# Prediction (panel-aware)
def predict_markov_panel(
    trans_matrix: np.ndarray,
    df_test: pd.DataFrame,
    initial_states: Optional[Dict[str, int]] = None,
    group_col: str = "join_key",
    time_col: str = "year",
    target_col: str = "target_is_disrupted_in_year",
    use_observed_prev: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict binary labels for a panel DataFrame using a fixed 2x2 transition matrix.

    For each group (port), predictions proceed in time order and are seeded with an
    initial previous-state provided in initial_states (fallback 0). The prediction
    for the next year is deterministic (argmax over P(next|current)). The flag
    use_observed_prev controls whether the true current state is used to seed the
    next prediction (operational mode) or whether the model's prediction is rolled forward
    (pure-forecast mode).

    Args:
        trans_matrix: 2x2 transition probability matrix.
        df_test: Panel DataFrame to predict (must contain group_col, time_col, and target_col).
        initial_states: Optional dict mapping group -> last observed state in training.
        group_col, time_col, target_col: column names.
        use_observed_prev: If True, use the true observed current state to seed next step.

    Returns:
        preds: 1D numpy array of predicted 0/1 values in the order of df_test grouped by (group_col, time_col).
        trues: 1D numpy array of true 0/1 values aligned with preds.
    """
    preds = []
    trues = []
    if initial_states is None:
        initial_states = {}

    # Iterate groups in deterministic order for reproducibility
    for _, grp in df_test.sort_values([group_col, time_col]).groupby(group_col, sort=True):
        grp_sorted = grp.sort_values(time_col)
        y_true = _validate_binary_series(grp_sorted[target_col].values)
        trues.extend(y_true.tolist())

        prev = int(initial_states.get(grp_sorted.iloc[0].get(group_col, ""), 0))
        # If there is an explicit mapping for this group, use it. Otherwise, fallback to default 0
        if grp_sorted.iloc[0].get(group_col, "") in initial_states:
            prev = int(initial_states[grp_sorted.iloc[0].get(group_col, "")])
        else:
            # If initial_states mapping uses keys equal to group values, use that.
            # Otherwise leave prev as 0 (safe fallback)
            prev = int(initial_states.get(grp_sorted.iloc[0].get(group_col, ""), 0))

        for i, true_val in enumerate(y_true):
            pred = int(np.argmax(trans_matrix[prev]))
            preds.append(pred)

            if use_observed_prev:
                prev = int(true_val)
            else:
                prev = int(pred)

    return np.asarray(preds, dtype=int), np.asarray(trues, dtype=int)


# Convenience: compute initial_states from training panel
def build_initial_states_from_train(
    df_train: pd.DataFrame,
    group_col: str = "join_key",
    time_col: str = "year",
    target_col: str = "target_is_disrupted_in_year"
) -> Dict[str, int]:
    """
    Return a mapping group -> last observed state (0/1) from the training panel.
    """
    init = {}
    for name, grp in df_train.groupby(group_col):
        last_row = grp.sort_values(time_col).iloc[-1]
        init[name] = int(bool(last_row[target_col]))
    return init

# Evaluation runner (panel-aware)
def run_markov_benchmark_panel(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    group_col: str = "join_key",
    time_col: str = "year",
    target_col: str = "target_is_disrupted_in_year",
    laplace: float = 1.0,
    use_observed_prev: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train and evaluate the Markov baseline on panel data. Returns metrics and diagnostics.

    Args:
        df_train, df_test: Training and test panels (DataFrames).
        laplace: Smoothing constant for training.
        use_observed_prev: Prediction mode flag.
        verbose: If True, print the transition matrix and summary.

    Returns:
        dict containing transition_matrix, accuracy, f1, confusion_matrix, n_test, and use_observed_prev.
    """
    trans_matrix = train_markov_chain_panel(
        df_train,
        group_col=group_col,
        time_col=time_col,
        target_col=target_col,
        laplace=laplace
    )

    initial_states = build_initial_states_from_train(df_train, group_col=group_col, time_col=time_col, target_col=target_col)

    preds, trues = predict_markov_panel(
        trans_matrix,
        df_test,
        initial_states=initial_states,
        group_col=group_col,
        time_col=time_col,
        target_col=target_col,
        use_observed_prev=use_observed_prev
    )

    acc = float(accuracy_score(trues, preds))
    f1 = float(f1_score(trues, preds, zero_division=0))
    cm = confusion_matrix(trues, preds)

    if verbose:
        print("\n> Markov Baseline (panel-aware) Results")
        print("-" * 60)
        print("Transition matrix P(next | current):")
        print(f"  [0 -> 0]: {trans_matrix[0,0]:.4f}   [0 -> 1]: {trans_matrix[0,1]:.4f}")
        print(f"  [1 -> 0]: {trans_matrix[1,0]:.4f}   [1 -> 1]: {trans_matrix[1,1]:.4f}")
        print(f"\nPrediction mode: {'use_observed_prev' if use_observed_prev else 'roll-forward (no observed prev)'}")
        print(f"Test samples: {len(trues)}  |  Accuracy: {acc:.4f}  |  F1: {f1:.4f}")
        print("Confusion matrix (rows=true, cols=pred):")
        print(cm)
        print("\nClassification report:")
        try:
            print(classification_report(trues, preds, zero_division=0))
        except Exception:
            pass

    return {
        "transition_matrix": trans_matrix,
        "accuracy": acc,
        "f1": f1,
        "confusion_matrix": cm,
        "n_test": int(len(trues)),
        "use_observed_prev": bool(use_observed_prev)
    }



# Backwards-compatible simple vector utilities
def train_markov_chain_vector(y_train: Iterable, laplace: float = 1.0) -> np.ndarray:
    """
    Train a transition matrix from a single long sequence.
    Done for quick experiments only (above is preferred).
    """
    y = _validate_binary_series(y_train)
    if y.size < 2:
        # return identity-ish matrix with smoothing
        counts = np.ones((2,2)) * laplace
    else:
        counts = np.zeros((2,2), dtype=float)
        for i in range(len(y) - 1):
            counts[y[i], y[i+1]] += 1.0
        counts += laplace
    probs = counts / (counts.sum(axis=1, keepdims=True) + EPS)
    return probs


def run_markov_benchmark_vector(y_train: Iterable, y_test: Iterable, laplace: float = 1.0, verbose: bool = True) -> Dict[str, Any]:
    """
    Legacy wrapper that trains from a single vector and predicts the test vector.
    This keeps compatibility with code that provides only arrays of labels.
    """
    trans_matrix = train_markov_chain_vector(y_train, laplace=laplace)

    preds = []
    prev = 0
    for _ in y_test:
        pred = int(np.argmax(trans_matrix[prev]))
        preds.append(pred)
        # Legacy loop uses true value as prev.
        # Only be used for quick diagnostics or where that assumption is valid.
        prev = int(_validate_binary_series([_])[0])  # converts the single test value

    trues = _validate_binary_series(y_test)
    acc = float(accuracy_score(trues, preds))
    f1 = float(f1_score(trues, preds, zero_division=0))
    cm = confusion_matrix(trues, preds)

    if verbose:
        print("\n> Markov Baseline (vector) Results")
        print("-" * 60)
        print("Transition matrix P(next | current):")
        print(trans_matrix)
        print(f"Accuracy: {acc:.4f}  |  F1: {f1:.4f}")
        print("Confusion matrix (rows=true, cols=pred):")
        print(cm)

    return {
        "transition_matrix": trans_matrix,
        "accuracy": acc,
        "f1": f1,
        "confusion_matrix": cm,
        "n_test": int(len(trues))
    }

# CLI unit test (synthetic)
if __name__ == "__main__":
    # Small synthetic demonstration to sanity-check
    print("Markov baseline self-check (synthetic example)...")

    # Build a tiny panel with two ports and simple dynamics
    df_train = pd.DataFrame([
        {"join_key": "port_a", "year": 2018, "target_is_disrupted_in_year": 0},
        {"join_key": "port_a", "year": 2019, "target_is_disrupted_in_year": 1},
        {"join_key": "port_a", "year": 2020, "target_is_disrupted_in_year": 1},
        {"join_key": "port_b", "year": 2018, "target_is_disrupted_in_year": 0},
        {"join_key": "port_b", "year": 2019, "target_is_disrupted_in_year": 0},
        {"join_key": "port_b", "year": 2020, "target_is_disrupted_in_year": 1},
    ])

    df_test = pd.DataFrame([
        {"join_key": "port_a", "year": 2021, "target_is_disrupted_in_year": 1},
        {"join_key": "port_b", "year": 2021, "target_is_disrupted_in_year": 0},
    ])

    result = run_markov_benchmark_panel(df_train, df_test, laplace=1.0, use_observed_prev=False, verbose=True)
    print("\nSelf-check complete. Result keys:", list(result.keys()))