"""
MARITIME DISRUPTION PROJECT - Multi-Model ML TRAINING PIPELINE

Production-ready training module that fits, validates, and compares a
collection of supervised classifiers on port-year features. It focuses on
robust, reproducible modelling by applying deterministic seeds, controlled
regularization, leakage detection, and time-aware cross-validation.

Key behaviours:
    - Accepts feature matrices and binary labels and coerces inputs to stable
      numeric numpy arrays for model compatibility.
    - Constructs moderately-clipped sample weights to stabilize training under
      class imbalance while avoiding extreme weight amplification.
    - Detects and removes explicit leakage features and implicit proxy features
      (highly correlated predictors with the target) to prevent data leakage.
    - Applies targeted regularization to reduce memorization:
        * Identity dropout: zeroes static/identifier-like columns (e.g., meta__,
          iso3, lat/lon) on a random subset of rows to force models to rely on
          dynamic signals.
        * Additive small Gaussian noise on numeric features for robustness.
    - Runs TimeSeriesSplit cross-validation (temporal folds), using sample
      weights where supported, to produce stable CV AUC estimates for model
      selection.
    - Trains a suite of classifiers (ExtraTrees, XGBoost, Random Forest, MLP,
      Logistic Regression, KNN) with consistent randomness and returns:
        * trained model objects (model.active_features_ = retained feature list)
        * per-model train/validation metrics and CV AUC
        * cleaned training/validation arrays and retained feature names
    - Emits clear, human-readable diagnostics (CV AUC, train/test AUC, log-loss,
      and identified leakage/proxy features) to guide downstream analysis.
"""
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
import traceback

RANDOM_SEED = 42

import xgboost as xgb
XGB_AVAILABLE = True


ET_AVAILABLE = True


# Utilities: small helper functions used throughout the training pipeline
# - _to_numpy: coerce DataFrame/array-like inputs into consistent float arrays
# - _compute_metrics: safe wrapper to compute accuracy, ROC AUC and log-loss
def _to_numpy(X):
    if isinstance(X, pd.DataFrame): return X.values.astype(float)
    elif isinstance(X, np.ndarray): return X.astype(float)
    else: return np.asarray(X).astype(float)

def _compute_metrics(model, X, y):
    try:
        preds = model.predict(X)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
            auc = float(roc_auc_score(y, probs))
            logloss = float(log_loss(y, probs, labels=[0, 1]))
        else:
            probs = preds
            auc = 0.5
            logloss = 999.0
            
        return {"accuracy": float(accuracy_score(y, preds)), "roc_auc": auc, "log_loss": logloss}
    except Exception as e:
        return {"accuracy": 0.0, "roc_auc": 0.5, "log_loss": 999.0}


# Sample weight generation (regularization)
# - Create inverse-frequency-based sample weights
# - Clip weights to a moderate range to avoid extreme influence on optimization
def generate_sample_weights_for_regularization(y_train: pd.Series, max_weight: float = 2.0, min_weight: float = 1.0) -> np.ndarray:
    n_samples = len(y_train)
    n_pos = np.sum(y_train)
    n_neg = n_samples - n_pos
    
    # Weights calculated based on inverse frequency
    weight_pos = n_samples / (2.0 * n_pos)
    weight_neg = n_samples / (2.0 * n_neg)
    
    weights = y_train.apply(lambda x: weight_pos if x == 1 else weight_neg).values
    weights = np.clip(weights, min_weight, max_weight)
    
    print(f"   [SWS] Sample weights generated. Range: [{min(weights):.2f}, {max(weights):.2f}]")
    return weights


# Time-series cross-validation runner
# - Performs TimeSeriesSplit CV
# - Fits the provided model function using training folds and evaluates AUC
# - Honors sample weights for algorithms that accept them; avoids weights for KNN
def run_time_series_cv(model_func, X_train, y_train, sample_weights, model_params, n_splits=5, is_knn=False):
    """ Runs TimeSeriesSplit CV and returns mean AUC."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    auc_scores = []
    
    X_train_df = pd.DataFrame(X_train)
    y_train_series = pd.Series(y_train)
    
    for train_index, val_index in tscv.split(X_train_df):
        X_tr, X_val = X_train[train_index], X_train[val_index]
        y_tr, y_val = y_train_series.iloc[train_index], y_train_series.iloc[val_index]
        w_tr = sample_weights[train_index] if sample_weights is not None else None
        
        if is_knn:
            # KNN is non-parametric and should not use sample_weights here
            model_cv = model_func(X_tr, y_tr, X_val=X_val, y_val=y_val, sample_weight=None) 
        else:
            # Use sample_weight for all others (ExtraTrees, XGB, RF, LR, MLP)
            model_cv = model_func(X_tr, y_tr, X_val=X_val, y_val=y_val, sample_weight=w_tr, **model_params)
        
        if hasattr(model_cv, "predict_proba"):
            probs = model_cv.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, probs)
            auc_scores.append(auc)
            
    return np.mean(auc_scores) if auc_scores else 0.5 


# Feature regularization helpers
# - apply_identity_dropout: randomly zero static identifier-like columns on a subset of rows to force models to use dynamic signals rather than memorized IDs.
# - add_noise: small gaussian noise to numeric columns to improve robustness.
# - diagnose_and_drop_highly_correlated: find and remove features that correlate very strongly with the label (likely proxies/leakage), reporting what is dropped.
def apply_identity_dropout(X, feature_names, dropout_prob=0.5):
    if not feature_names or X.shape[1] != len(feature_names): return X
    
    static_indices = []
    for i, name in enumerate(feature_names):
        ln = name.lower()
        if any(k in ln for k in ["meta__", "country__", "iso3", "lat", "lon", "region", "un_locode"]):
            static_indices.append(i)
            
    if not static_indices: 
        print("   [WARNING] No static features found for dropout, model relying on clean dynamic data.")
        return X

    np.random.seed(RANDOM_SEED)
    
    X_aug = X.copy()
    mask = np.random.rand(X.shape[0]) < dropout_prob
    X_aug[np.ix_(mask, static_indices)] = 0.0
    return X_aug

def add_noise(X, noise_level=0.05):
    np.random.seed(RANDOM_SEED)
    sigma = noise_level * np.std(X, axis=0)
    sigma[sigma == 0] = 0.0001
    noise = np.random.normal(0, sigma, X.shape)
    return X + noise

def diagnose_and_drop_highly_correlated(X_train: np.ndarray, y_train: pd.Series, feature_names: List[str], threshold: float = 0.80):
    if X_train.shape[1] == 0:
        return X_train, feature_names, []

    X_df = pd.DataFrame(X_train, columns=feature_names)
    y_series = y_train.reset_index(drop=True)
    corr_series = X_df.apply(lambda x: x.corr(y_series)).abs()
    highly_correlated_features = corr_series[corr_series >= threshold].index.tolist()
    leakage_candidates = highly_correlated_features 

    indices_to_drop = [feature_names.index(f) for f in leakage_candidates if f in feature_names]
    
    if indices_to_drop:
        print(f"\n   [DIAGNOSTIC] Found {len(indices_to_drop)} features with |Corr| >= {threshold} (FATAL PROXIES):")
        for feature in leakage_candidates:
            print(f"      - FOUND & DROPPING: {feature} (Corr: {corr_series[feature]:.4f})")
            
        X_train_clean = np.delete(X_train, indices_to_drop, axis=1)
        kept_names = [f for f in feature_names if f not in leakage_candidates]
        return X_train_clean, kept_names, leakage_candidates
    
    print(f"\n   [DIAGNOSTIC] No highly correlated features found (Max Corr < {threshold}).")
    return X_train, feature_names, []


# Model training wrappers
# - Each wrapper fits a model and accepts sample_weight where applicable.
def train_extratrees(X_train, y_train, X_val=None, y_val=None, sample_weight=None, **kwargs):
    """
    ExtraTrees wrapper (robust, simple, no extra deps). Accepts sample_weight.
    """
    if not ET_AVAILABLE:
        raise ImportError("ExtraTreesClassifier missing")

    params = {}
    params['n_estimators'] = int(kwargs.pop('n_estimators', 200))
    if 'max_features' in kwargs:
        params['max_features'] = kwargs.pop('max_features')
    if 'max_depth' in kwargs:
        params['max_depth'] = kwargs.pop('max_depth')
    params['random_state'] = RANDOM_SEED
    params['n_jobs'] = -1

    model = ExtraTreesClassifier(**params)

    # Fit with sample_weight if supported; fallback otherwise
    try:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    except TypeError:
        model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, X_val=None, y_val=None, sample_weight=None, **kwargs):
    if not XGB_AVAILABLE: raise ImportError("XGBoost missing")
    # Removed scale_pos_weight from params to fix log loss. Will use sample_weight.
    if 'scale_pos_weight' in kwargs: kwargs.pop('scale_pos_weight')
    model = xgb.XGBClassifier(
        n_estimators=100, n_jobs=-1, random_state=RANDOM_SEED, **kwargs
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model



# Random forest / ensemble wrapper
# - Trained without class_weight; sample_weight is provided at fit time for calibration
def train_random_forest(X_train, y_train, X_val=None, y_val=None, sample_weight=None, **kwargs):
    # Reverted class_weight back to None to rely on sample_weight and fix calibration
    model = RandomForestClassifier(
        n_estimators=150, class_weight=None, n_jobs=-1, random_state=RANDOM_SEED, **kwargs
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


# MLP, LR and KNN wrappers
# - MLP and LR use sample_weight where supported; KNN is trained normally.
def train_mlp(X_train, y_train, X_val=None, y_val=None, sample_weight=None, **kwargs):
    # Fix remains: Removed conflicting learning_rate_init
    model = MLPClassifier(
        hidden_layer_sizes=(16, 8), max_iter=500, random_state=RANDOM_SEED, **kwargs
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model

def train_lr(X_train, y_train, X_val=None, y_val=None, sample_weight=None, **kwargs):
    # Fix remains: Removed conflicting C parameter. Reverted class_weight back to None.
    model = LogisticRegression(
        penalty='l2', class_weight=None, max_iter=1000, random_state=RANDOM_SEED, **kwargs
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model

def train_knn(X_train, y_train, X_val=None, y_val=None, sample_weight=None, **kwargs):
    model = KNeighborsClassifier(n_neighbors=500)
    model.fit(X_train, y_train)
    return model



# Pipeline execution helpers
# - identify_and_drop_leakage: remove explicit answer-key columns (disrupted_flag/target-like)
def identify_and_drop_leakage(X_train, X_val, feature_names):
    if not feature_names: return X_train, X_val, [], []
    indices = []
    dropped = []
    for i, name in enumerate(feature_names):
        ln = name.lower()
        if ("disrupted_flag" in ln) or ("target_" in ln) or ("_dis" in ln and "events__" in ln):
            indices.append(i)
            dropped.append(name)
    
    if indices:
        print(f"\n   [FEATURE CONTROL] Removed {len(indices)} Leakage features (Answer Keys).")
        X_tr = np.delete(X_train, indices, axis=1)
        X_vl = np.delete(X_val, indices, axis=1) if X_val is not None else None
        kept = [feature_names[i] for i in range(len(feature_names)) if i not in indices]
        return X_tr, X_vl, kept, dropped
    return X_train, X_val, feature_names, []


# Main training orchestration
# - Prepares arrays, applies diagnostics and regularization, runs CV and final fits.
# - Returns dictionary of trained models and metrics plus cleaned arrays and feature list.
def train_all_models(X_train, y_train, X_val=None, y_val=None, include_xgb: bool = True, feature_names=None, **kwargs):
    
    X_tr_np = _to_numpy(X_train)
    X_val_np = _to_numpy(X_val) if X_val is not None else None
    y_train_discrete = pd.Series(y_train).astype(int)
    
    # Updated model_configs: 'et' replaces previous 'hgb' entry
    model_configs = {
        'et': {'n_estimators': 200, 'max_features': 0.1, 'max_depth': 3},
        'xgb': {'max_depth': 2, 'learning_rate': 0.05, 'colsample_bytree': 0.1, 'subsample': 0.5, 'reg_alpha': 50.0, 'reg_lambda': 50.0},
        'rf': {'max_depth': 3, 'min_samples_leaf': 500, 'max_features': 0.1}, 
        'mlp': {'alpha': 5.0, 'learning_rate_init': 0.0001}, 
        'lr': {'C': 0.001}, 
        'knn': {'n_neighbors': 500},
    }
    
    results = {"models": {}, "metrics": {}}
    
    
    # 2. Generate Sample Weights (SWS) - Now with moderate clipping [1.0, 2.0]
    sample_weights = generate_sample_weights_for_regularization(y_train_discrete) 

    # 3. Initial Drop: remove explicit leakage columns
    X_tr_cl, X_val_cl, kept_names, initial_dropped = identify_and_drop_leakage(X_tr_np, X_val_np, feature_names)
    
    # 4. DIAGNOSTIC DROP: Identify and remove hidden highly-correlated features
    X_tr_diag, kept_names_diag, diag_dropped = diagnose_and_drop_highly_correlated(X_tr_cl, y_train_discrete, kept_names, threshold=0.80)
    
    # 5. If a hidden feature was found, update the training data AND the validation data
    if diag_dropped:
        print(f"\n   [CRITICAL FIX APPLIED] Dropped {len(diag_dropped)} features found by correlation diagnostic.")
        
        indices_to_drop_in_val = [kept_names.index(f) for f in diag_dropped if f in kept_names]
        
        if X_val_cl is not None:
            X_val_cl = np.delete(X_val_cl, indices_to_drop_in_val, axis=1)
        
        X_tr_cl = X_tr_diag
        kept_names = kept_names_diag
    
    # 6. APPLY REGULARIZATION (The Pattern Enforcer)
    X_tr_aug = X_tr_cl.copy()
    
    # Apply Identity Dropout to break static reliance
    print("   [TRAINING] Applying 'Identity Dropout' (50%) to break static ID memorization...")
    X_tr_aug = apply_identity_dropout(X_tr_aug, kept_names, dropout_prob=0.5)
    
    # Add noise to destabilize remaining numerical features
    X_tr_aug = add_noise(X_tr_aug, noise_level=0.05)


    # Priority list: models to train (ExtraTrees replaces previous HistGradientBoosting entry)
    funcs = [
        ("et", train_extratrees) if ET_AVAILABLE else None,
        ("xgb", train_xgboost) if include_xgb and XGB_AVAILABLE else None,
        ("rf", train_random_forest), 
        ("mlp", train_mlp),
        ("lr", train_lr), 
        ("knn", train_knn), 
    ]
    funcs = [f for f in funcs if f is not None]

    print("\n" + "="*100)
    print(f" {'MODEL':<10} | {'TRAIN AUC':<10} | {'CV AUC (5-Fold)':<15} | {'TEST AUC':<10} | {'LOG LOSS':<10} | {'GAP (T-Te)':<10} | {'STATUS':<10}")
    print("-" * 100)

    for name, func in funcs:
        try:
            model_params = model_configs.get(name, {})
            
            # 1. Run CV
            is_knn = (name == 'knn')
            
            # All models (except KNN) now rely on the moderately-clipped sample_weights
            cv_sample_weights = None if is_knn else sample_weights
            
            cv_model_params = model_configs.get(name, {})
            
            cv_auc = run_time_series_cv(func, X_tr_aug, y_train_discrete, cv_sample_weights, cv_model_params, n_splits=5, is_knn=is_knn)

            # 2. Train Final Model
            if is_knn:
                # KNN does not use sample_weight
                model = func(X_tr_cl, y_train_discrete, X_val=X_val_cl, y_val=y_val)
            else:
                # All other models use the moderately-clipped sample_weights
                model = func(X_tr_aug, y_train_discrete, X_val=X_val_cl, y_val=y_val, sample_weight=sample_weights, **model_params)

            model.active_features_ = kept_names # Save final feature list to model
            
            # 3. Evaluate
            # Evaluation uses the CLEANED, non-augmented data
            met_tr = _compute_metrics(model, X_tr_cl, y_train_discrete)
            met_te = _compute_metrics(model, X_val_cl, y_val) if X_val_cl is not None else {}
            
            sc_tr = met_tr.get("roc_auc", 0.0)
            sc_te = met_te.get("roc_auc", 0.0)
            logloss_te = met_te.get("log_loss", 999.0)
            
            gap_t_te = sc_tr - sc_te
            
            print(f" {name:<10} | {sc_tr:.4f}     | {cv_auc:.4f}            | {sc_te:.4f}     | {logloss_te:.4f}    | {gap_t_te:+.4f}     | (v) Done")
            
            results["models"][name] = model
            results["metrics"][name] = {"train": met_tr, "val": met_te, "cv_auc": cv_auc}
        except Exception as e:
            print(f" {name:<10} | {'ERROR':<10} | {'N/A':<15}     | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | (x) Failed")
            traceback.print_exc()

    print("="*100 + "\n")
    
    # Return cleaned arrays and feature names
    return results, X_tr_cl, X_val_cl, kept_names