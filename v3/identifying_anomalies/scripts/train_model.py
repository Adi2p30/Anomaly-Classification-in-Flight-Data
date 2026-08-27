"""Train three anomaly-window detectors on the physics-feature table:

- a Random Forest (as in Initial_Run),
- a HistGradientBoostingClassifier with group-aware isotonic calibration
  (new in v3, a stronger gradient-boosted alternative), and
- a classical single-feature (lsq_residual_energy) threshold baseline
  (the matched-filter / energy-detector answer to the same question),

using a GroupShuffleSplit grouped by base_signal_id so the same (f0, phase)
combination never appears in both train and test. On top of that single
split (whose artifacts everything downstream -- eval CSVs, plots, the
interactive viewer -- consumes), a GroupKFold cross-validation pass gives
mean +/- std metrics per model for an honest generalization claim, since 28
base signals is still a small basis for a single point estimate."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "generating_data" / "scripts"))
import config
import features as feat


def fit_threshold(scores, labels):
    """Pick the score threshold that maximizes Youden's J (tpr - fpr) on
    the given (training or calibration) split. Generic over any real-valued
    score -- used for both the baseline's lsq_residual_energy and the
    calibrated HGB's predicted probability."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    return float(thresholds[best_idx])


def score_predictions(y_true, y_pred, y_score):
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def fit_calibrated_hgb(X_train, y_train, groups_train):
    """Fit HistGradientBoostingClassifier on a fit sub-split, then isotonic-
    calibrate on a disjoint, group-aware calibration sub-split carved out of
    the same training groups (plain CalibratedClassifierCV(cv=k) is not
    group-aware and would leak windows from the same file across the
    fit/calibrate split). Returns (calibrated_model, threshold), the
    threshold chosen via Youden's J on the calibration split only -- never
    on test."""
    calib_splitter = GroupShuffleSplit(
        n_splits=1, test_size=config.CALIB_HOLDOUT_FRACTION, random_state=config.RANDOM_SEED
    )
    fit_sub_idx, calib_sub_idx = next(calib_splitter.split(X_train, y_train, groups_train))

    hgb = HistGradientBoostingClassifier(**config.HGB_PARAMS)
    hgb.fit(X_train[fit_sub_idx], y_train[fit_sub_idx])

    calibrated_hgb = CalibratedClassifierCV(estimator=FrozenEstimator(hgb), method="isotonic")
    calibrated_hgb.fit(X_train[calib_sub_idx], y_train[calib_sub_idx])

    calib_proba = calibrated_hgb.predict_proba(X_train[calib_sub_idx])[:, 1]
    threshold = fit_threshold(calib_proba, y_train[calib_sub_idx])
    return calibrated_hgb, threshold, groups_train[calib_sub_idx]


def cv_fold_metrics(X_train, y_train, groups_train, X_test, y_test):
    """Fit RF, calibrated HGB, and the lsq-residual baseline on one
    GroupKFold fold's training split, score on its test split. Used only
    for the cross-validated confidence-interval metrics -- discards the
    fitted models, unlike the primary split below which keeps them."""
    rf = RandomForestClassifier(**config.RF_PARAMS)
    rf.fit(X_train, y_train)
    rf_metrics = score_predictions(y_test, rf.predict(X_test), rf.predict_proba(X_test)[:, 1])

    calibrated_hgb, hgb_threshold, _ = fit_calibrated_hgb(X_train, y_train, groups_train)
    y_proba_hgb = calibrated_hgb.predict_proba(X_test)[:, 1]
    hgb_metrics = score_predictions(y_test, (y_proba_hgb >= hgb_threshold).astype(int), y_proba_hgb)

    residual_col = feat.FEATURE_COLUMNS.index("lsq_residual_energy")
    baseline_threshold = fit_threshold(X_train[:, residual_col], y_train)
    y_pred_baseline = (X_test[:, residual_col] >= baseline_threshold).astype(int)
    baseline_metrics = score_predictions(y_test, y_pred_baseline, X_test[:, residual_col])

    return {"random_forest": rf_metrics, "hist_gradient_boosting": hgb_metrics, "baseline": baseline_metrics}


def main():
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.FEATURES_CSV)

    X = df[feat.FEATURE_COLUMNS].to_numpy()
    y = df["is_anomalous"].to_numpy()
    groups = df["base_signal_id"].to_numpy()

    # --- primary split: produces the artifacts everything downstream uses ---
    splitter = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]

    n_train_groups = len(np.unique(groups_train))
    n_test_groups = len(np.unique(groups[test_idx]))
    print(f"train: {len(train_idx)} windows from {n_train_groups} base signals")
    print(f"test:  {len(test_idx)} windows from {n_test_groups} base signals")

    # --- Random Forest on physics-based features ---
    rf = RandomForestClassifier(**config.RF_PARAMS)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    rf_metrics = score_predictions(y_test, y_pred_rf, y_proba_rf)
    rf_metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred_rf).tolist()
    feature_importances = dict(zip(feat.FEATURE_COLUMNS, rf.feature_importances_.tolist()))

    # --- HistGradientBoosting, group-aware calibrated ---
    calibrated_hgb, hgb_threshold, calib_groups = fit_calibrated_hgb(X_train, y_train, groups_train)
    y_proba_hgb = calibrated_hgb.predict_proba(X_test)[:, 1]
    y_pred_hgb = (y_proba_hgb >= hgb_threshold).astype(int)
    hgb_metrics = score_predictions(y_test, y_pred_hgb, y_proba_hgb)
    hgb_metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred_hgb).tolist()

    perm = permutation_importance(
        calibrated_hgb, X_test, y_test, n_repeats=10, random_state=config.RANDOM_SEED
    )
    hgb_feature_importances = dict(zip(feat.FEATURE_COLUMNS, perm.importances_mean.tolist()))

    # --- classical least-squares-residual threshold baseline ---
    residual_col = feat.FEATURE_COLUMNS.index("lsq_residual_energy")
    baseline_threshold = fit_threshold(X_train[:, residual_col], y_train)
    y_pred_baseline = (X_test[:, residual_col] >= baseline_threshold).astype(int)
    baseline_metrics = score_predictions(y_test, y_pred_baseline, X_test[:, residual_col])
    baseline_metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred_baseline).tolist()

    print("\nRandom Forest:", json.dumps(rf_metrics, indent=2))
    print("\nHistGradientBoosting:", json.dumps(hgb_metrics, indent=2))
    print("\nBaseline (lsq residual threshold):", json.dumps(baseline_metrics, indent=2))

    # --- nested GroupKFold for confidence-interval metrics (separate from
    # the artifact-producing split above: refits RF/HGB/baseline per fold) ---
    print(f"\nrunning {config.N_CV_FOLDS}-fold GroupKFold cross-validation for confidence intervals...")
    cv_raw = {"random_forest": [], "hist_gradient_boosting": [], "baseline": []}
    cv_splitter = GroupKFold(n_splits=config.N_CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    for fold_train_idx, fold_test_idx in cv_splitter.split(X, y, groups):
        fold_metrics = cv_fold_metrics(
            X[fold_train_idx], y[fold_train_idx], groups[fold_train_idx],
            X[fold_test_idx], y[fold_test_idx],
        )
        for name, m in fold_metrics.items():
            cv_raw[name].append(m)

    cv_metrics = {
        name: {
            metric: {
                "mean": float(np.nanmean([m[metric] for m in fold_list])),
                "std": float(np.nanstd([m[metric] for m in fold_list])),
            }
            for metric in ["precision", "recall", "f1", "roc_auc"]
        }
        for name, fold_list in cv_raw.items()
    }
    print("\nCross-validated metrics (mean +/- std over "
          f"{config.N_CV_FOLDS} folds):", json.dumps(cv_metrics, indent=2))

    joblib.dump(rf, config.MODEL_PATH)
    joblib.dump(calibrated_hgb, config.HGB_MODEL_PATH)
    with open(config.BASELINE_THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": baseline_threshold, "feature": "lsq_residual_energy"}, f, indent=2)
    with open(config.HGB_THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": hgb_threshold, "feature": "hgb_proba"}, f, indent=2)
    with open(config.METRICS_PATH, "w") as f:
        json.dump(
            {
                "random_forest": rf_metrics,
                "hist_gradient_boosting": hgb_metrics,
                "baseline": baseline_metrics,
                "feature_importances": feature_importances,
                "hgb_feature_importances": hgb_feature_importances,
                "cv_metrics": cv_metrics,
                "train_base_signal_ids": np.unique(groups_train).tolist(),
                "calib_base_signal_ids": np.unique(calib_groups).tolist(),
                "test_base_signal_ids": np.unique(groups[test_idx]).tolist(),
            },
            f,
            indent=2,
        )

    print(f"\nwrote {config.MODEL_PATH}")
    print(f"wrote {config.HGB_MODEL_PATH}")
    print(f"wrote {config.BASELINE_THRESHOLD_PATH}")
    print(f"wrote {config.HGB_THRESHOLD_PATH}")
    print(f"wrote {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
