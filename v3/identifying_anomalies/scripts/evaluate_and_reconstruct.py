"""Evaluate the trained models on the held-out test files: reconstruct a
contiguous predicted anomaly interval per file from the sliding-window
predictions, compare it to the ground-truth window (IoU + boundary error in
seconds), and summarize detection quality as a function of anomaly type and
severity (the difficulty sweep, now three anomaly types instead of one).
Outputs are consumed by make_plots.py.

Reconstruction is run twice per file on the Random Forest's window
probabilities: once with no smoothing (Initial_Run-equivalent, ->
eval_per_file_raw.csv) and once with v3's median-filter + hysteresis
smoothing (-> eval_per_file.csv, the one plots/interactive_viewer.py
consume) -- a real before/after comparison rather than a claimed one, since
v3's non-stationary/noisy baseline makes spurious control-file detections a
real risk that Initial_Run's noise-free pipeline never had to worry about.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "generating_data" / "scripts"))
import config
import features as feat
from reconstruct import reconstruct_window


def interval_iou(pred_start, pred_end, true_start, true_end):
    has_pred = pred_start is not None
    has_true = true_start is not None and true_start >= 0
    if not has_pred and not has_true:
        return 1.0
    if not has_pred or not has_true:
        return 0.0
    overlap = max(0, min(pred_end, true_end) - max(pred_start, true_start))
    union = max(pred_end, true_end) - min(pred_start, true_start)
    return overlap / union if union > 0 else 1.0


def reconstruct_per_file(test_df, labels_df, median_kernel, hysteresis):
    rows = []
    for filename, file_windows in test_df.groupby("filename"):
        file_windows = file_windows.sort_values("window_start")
        pred_start, pred_end = reconstruct_window(
            file_windows["window_start"].to_numpy(),
            file_windows["window_end"].to_numpy(),
            file_windows["y_proba_rf"].to_numpy(),
            config.N_SAMPLES,
            median_kernel=median_kernel,
            hysteresis=hysteresis,
        )

        label_row = labels_df.loc[filename]
        true_start = int(label_row["start_idx"])
        true_end = int(label_row["end_idx"])

        iou = interval_iou(pred_start, pred_end, true_start, true_end)
        start_error_s = (
            abs(pred_start - true_start) / config.FS
            if pred_start is not None and true_start >= 0 else np.nan
        )
        end_error_s = (
            abs(pred_end - true_end) / config.FS
            if pred_end is not None and true_end >= 0 else np.nan
        )

        rows.append(
            {
                "filename": filename,
                "base_signal_id": file_windows["base_signal_id"].iloc[0],
                "anomaly_type": file_windows["anomaly_type"].iloc[0],
                "severity": file_windows["severity"].iloc[0],
                "true_start_s": true_start / config.FS if true_start >= 0 else np.nan,
                "true_end_s": true_end / config.FS if true_end >= 0 else np.nan,
                "pred_start_s": pred_start / config.FS if pred_start is not None else np.nan,
                "pred_end_s": pred_end / config.FS if pred_end is not None else np.nan,
                "iou": iou,
                "start_error_s": start_error_s,
                "end_error_s": end_error_s,
            }
        )
    return pd.DataFrame(rows)


def control_false_positive_rate(per_file_df):
    controls = per_file_df[per_file_df["anomaly_type"] == "none"]
    if len(controls) == 0:
        return float("nan")
    return float((~controls["pred_start_s"].isna()).mean())


def main():
    df = pd.read_csv(config.FEATURES_CSV)
    labels_df = pd.read_csv(config.LABELS_CSV).set_index("filename")

    rf_model = joblib.load(config.MODEL_PATH)
    hgb_model = joblib.load(config.HGB_MODEL_PATH)
    with open(config.BASELINE_THRESHOLD_PATH) as f:
        baseline_threshold = json.load(f)["threshold"]
    with open(config.HGB_THRESHOLD_PATH) as f:
        hgb_threshold = json.load(f)["threshold"]
    with open(config.METRICS_PATH) as f:
        metrics = json.load(f)
    test_ids = set(metrics["test_base_signal_ids"])

    test_df = df[df["base_signal_id"].isin(test_ids)].copy()
    X_test = test_df[feat.FEATURE_COLUMNS].to_numpy()
    test_df["y_pred_rf"] = rf_model.predict(X_test)
    test_df["y_proba_rf"] = rf_model.predict_proba(X_test)[:, 1]
    test_df["y_proba_hgb"] = hgb_model.predict_proba(X_test)[:, 1]
    test_df["y_pred_hgb"] = (test_df["y_proba_hgb"] >= hgb_threshold).astype(int)
    test_df["y_pred_baseline"] = (test_df["lsq_residual_energy"] >= baseline_threshold).astype(int)

    test_df.to_csv(config.EVAL_TEST_PREDICTIONS_CSV, index=False)

    # --- per-file reconstruction, raw vs. smoothed (RF probabilities) ---
    per_file_raw_df = reconstruct_per_file(test_df, labels_df, median_kernel=None, hysteresis=None)
    per_file_df = reconstruct_per_file(
        test_df, labels_df,
        median_kernel=config.RECONSTRUCT_MEDIAN_KERNEL, hysteresis=config.RECONSTRUCT_HYSTERESIS,
    )
    per_file_raw_df.to_csv(config.EVAL_PER_FILE_RAW_CSV, index=False)
    per_file_df.to_csv(config.EVAL_PER_FILE_CSV, index=False)

    raw_fp_rate = control_false_positive_rate(per_file_raw_df)
    smoothed_fp_rate = control_false_positive_rate(per_file_df)

    # --- per-type/per-severity sweep: F1 (window-level, RF & HGB) and mean IoU ---
    sweep_rows = []
    for (anomaly_type, severity), group in test_df.groupby(["anomaly_type", "severity"]):
        f1_rf = f1_score(group["is_anomalous"], group["y_pred_rf"], zero_division=0)
        f1_hgb = f1_score(group["is_anomalous"], group["y_pred_hgb"], zero_division=0)
        iou_group = per_file_df[
            (per_file_df["anomaly_type"] == anomaly_type) & (per_file_df["severity"] == severity)
        ]
        sweep_rows.append(
            {
                "anomaly_type": anomaly_type,
                "severity": severity,
                "f1_window_level_rf": f1_rf,
                "f1_window_level_hgb": f1_hgb,
                "mean_iou": iou_group["iou"].mean(),
                "n_files": len(iou_group),
            }
        )
    sweep_df = pd.DataFrame(sweep_rows).sort_values(["anomaly_type", "severity"])
    sweep_df.to_csv(config.EVAL_SWEEP_BY_TYPE_CSV, index=False)

    print(per_file_df[["filename", "anomaly_type", "severity", "iou", "start_error_s", "end_error_s"]]
          .to_string(index=False))
    print("\nper-type/severity sweep:")
    print(sweep_df.to_string(index=False))
    print(f"\ncontrol false-positive rate: raw={raw_fp_rate:.3f}, smoothed={smoothed_fp_rate:.3f}")
    print(f"\nwrote {config.EVAL_TEST_PREDICTIONS_CSV}")
    print(f"wrote {config.EVAL_PER_FILE_CSV}")
    print(f"wrote {config.EVAL_PER_FILE_RAW_CSV}")
    print(f"wrote {config.EVAL_SWEEP_BY_TYPE_CSV}")


if __name__ == "__main__":
    main()
