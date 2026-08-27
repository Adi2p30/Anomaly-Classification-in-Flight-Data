"""Evaluate the trained model on the held-out test files: reconstruct a
contiguous predicted anomaly interval per file from the sliding-window
predictions, compare it to the ground-truth window (IoU + boundary error in
seconds), and summarize detection quality as a function of the injected
anomaly's amplitude ratio (the difficulty sweep). Outputs are consumed by
make_plots.py."""

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


def main():
    df = pd.read_csv(config.FEATURES_CSV)
    labels_df = pd.read_csv(config.LABELS_CSV).set_index("filename")
    model = joblib.load(config.MODEL_PATH)
    with open(config.BASELINE_THRESHOLD_PATH) as f:
        threshold = json.load(f)["threshold"]
    with open(config.METRICS_PATH) as f:
        metrics = json.load(f)
    test_ids = set(metrics["test_base_signal_ids"])

    test_df = df[df["base_signal_id"].isin(test_ids)].copy()
    X_test = test_df[feat.FEATURE_COLUMNS].to_numpy()
    test_df["y_pred_rf"] = model.predict(X_test)
    test_df["y_proba_rf"] = model.predict_proba(X_test)[:, 1]
    test_df["y_pred_baseline"] = (test_df["lsq_residual_energy"] >= threshold).astype(int)

    test_df.to_csv(config.MODEL_DIR / "eval_test_predictions.csv", index=False)

    # --- per-file reconstruction ---
    # Interpolate each window's anomaly probability to sample resolution and
    # take the 0.5-crossing as the boundary, rather than snapping to the
    # window_stride grid -- see reconstruct.py for why this is more precise.
    per_file_rows = []
    for filename, file_windows in test_df.groupby("filename"):
        file_windows = file_windows.sort_values("window_start")
        pred_start, pred_end = reconstruct_window(
            file_windows["window_start"].to_numpy(),
            file_windows["window_end"].to_numpy(),
            file_windows["y_proba_rf"].to_numpy(),
            config.N_SAMPLES,
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

        per_file_rows.append(
            {
                "filename": filename,
                "base_signal_id": file_windows["base_signal_id"].iloc[0],
                "amp_ratio": file_windows["amp_ratio"].iloc[0],
                "true_start_s": true_start / config.FS if true_start >= 0 else np.nan,
                "true_end_s": true_end / config.FS if true_end >= 0 else np.nan,
                "pred_start_s": pred_start / config.FS if pred_start is not None else np.nan,
                "pred_end_s": pred_end / config.FS if pred_end is not None else np.nan,
                "iou": iou,
                "start_error_s": start_error_s,
                "end_error_s": end_error_s,
            }
        )

    per_file_df = pd.DataFrame(per_file_rows)
    per_file_df.to_csv(config.MODEL_DIR / "eval_per_file.csv", index=False)

    # --- difficulty sweep: F1 (window-level) and mean IoU by amplitude ratio ---
    sweep_rows = []
    for amp_ratio, group in test_df.groupby("amp_ratio"):
        f1 = f1_score(group["is_anomalous"], group["y_pred_rf"], zero_division=0)
        iou_group = per_file_df[per_file_df["amp_ratio"] == amp_ratio]
        sweep_rows.append(
            {
                "amp_ratio": amp_ratio,
                "f1_window_level": f1,
                "mean_iou": iou_group["iou"].mean(),
                "n_files": len(iou_group),
            }
        )
    sweep_df = pd.DataFrame(sweep_rows).sort_values("amp_ratio")
    sweep_df.to_csv(config.MODEL_DIR / "eval_amp_sweep.csv", index=False)

    print(per_file_df[["filename", "amp_ratio", "iou", "start_error_s", "end_error_s"]].to_string(index=False))
    print("\namplitude-ratio sweep:")
    print(sweep_df.to_string(index=False))
    print(f"\nwrote {config.MODEL_DIR / 'eval_test_predictions.csv'}")
    print(f"wrote {config.MODEL_DIR / 'eval_per_file.csv'}")
    print(f"wrote {config.MODEL_DIR / 'eval_amp_sweep.csv'}")


if __name__ == "__main__":
    main()
