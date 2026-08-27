"""Train the physics-feature Random Forest classifier plus a classical
least-squares-residual threshold baseline (the matched-filter / energy-
detector answer to the same question), using a GroupShuffleSplit grouped by
base_signal_id so the same (f0, phase) combination never appears in both
train and test."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "generating_data" / "scripts"))
import config
import features as feat


def fit_baseline_threshold(residuals, labels):
    """Pick the residual-energy threshold that maximizes Youden's J
    (tpr - fpr) on the training split."""
    fpr, tpr, thresholds = roc_curve(labels, residuals)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    return float(thresholds[best_idx])


def main():
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.FEATURES_CSV)

    X = df[feat.FEATURE_COLUMNS].to_numpy()
    y = df["is_anomalous"].to_numpy()
    groups = df["base_signal_id"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    n_train_groups = len(np.unique(groups[train_idx]))
    n_test_groups = len(np.unique(groups[test_idx]))
    print(f"train: {len(train_idx)} windows from {n_train_groups} base signals")
    print(f"test:  {len(test_idx)} windows from {n_test_groups} base signals")

    # --- Random Forest on physics-based features ---
    model = RandomForestClassifier(**config.RF_PARAMS)
    model.fit(X_train, y_train)
    y_pred_rf = model.predict(X_test)
    y_proba_rf = model.predict_proba(X_test)[:, 1]

    rf_metrics = {
        "precision": float(precision_score(y_test, y_pred_rf)),
        "recall": float(recall_score(y_test, y_pred_rf)),
        "f1": float(f1_score(y_test, y_pred_rf)),
        "roc_auc": float(roc_auc_score(y_test, y_proba_rf)),
        "confusion_matrix": confusion_matrix(y_test, y_pred_rf).tolist(),
    }
    feature_importances = dict(zip(feat.FEATURE_COLUMNS, model.feature_importances_.tolist()))

    # --- classical least-squares-residual threshold baseline ---
    residual_col = feat.FEATURE_COLUMNS.index("lsq_residual_energy")
    threshold = fit_baseline_threshold(X_train[:, residual_col], y_train)
    y_pred_baseline = (X_test[:, residual_col] >= threshold).astype(int)

    baseline_metrics = {
        "threshold": threshold,
        "precision": float(precision_score(y_test, y_pred_baseline)),
        "recall": float(recall_score(y_test, y_pred_baseline)),
        "f1": float(f1_score(y_test, y_pred_baseline)),
        "roc_auc": float(roc_auc_score(y_test, X_test[:, residual_col])),
        "confusion_matrix": confusion_matrix(y_test, y_pred_baseline).tolist(),
    }

    print("\nRandom Forest:", json.dumps(rf_metrics, indent=2))
    print("\nBaseline (lsq residual threshold):", json.dumps(baseline_metrics, indent=2))

    joblib.dump(model, config.MODEL_PATH)
    with open(config.BASELINE_THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": threshold, "feature": "lsq_residual_energy"}, f, indent=2)
    with open(config.METRICS_PATH, "w") as f:
        json.dump(
            {
                "random_forest": rf_metrics,
                "baseline": baseline_metrics,
                "feature_importances": feature_importances,
                "train_base_signal_ids": np.unique(groups[train_idx]).tolist(),
                "test_base_signal_ids": np.unique(groups[test_idx]).tolist(),
            },
            f,
            indent=2,
        )

    print(f"\nwrote {config.MODEL_PATH}")
    print(f"wrote {config.BASELINE_THRESHOLD_PATH}")
    print(f"wrote {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
