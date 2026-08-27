"""Build the sliding-window feature table used to train/evaluate the
classifier. f0 is estimated from each whole record (not read from the
generation metadata) so the pipeline stays honest about what a real
detector would have access to; the true f0 from labels.csv is only used
here to log the estimator's error as a sanity check."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "generating_data" / "scripts"))
import config
import features as feat


def build_rows_for_file(row, config_mod):
    path = config_mod.ANOMALY_DATA_DIR / row["filename"]
    df = pd.read_csv(path)
    values = df["value"].to_numpy()
    fs = config_mod.FS

    f0_hat = feat.estimate_f0(values, fs)
    f0_true = row["f0"]
    f0_error = abs(f0_hat - f0_true)

    windows = feat.sliding_windows(len(values), config_mod.WINDOW_LEN, config_mod.WINDOW_STRIDE)
    rows = []
    for start, end in windows:
        window_values = values[start:end]
        window_feats = feat.extract_window_features(window_values, fs, f0_hat)
        is_anomalous = feat.label_window(start, end, row["start_idx"], row["end_idx"])

        rows.append(
            {
                "filename": row["filename"],
                "base_signal_id": row["base_signal_id"],
                "window_start": start,
                "window_end": end,
                "f0_hat": f0_hat,
                "f0_true": f0_true,
                "f0_error": f0_error,
                "amp_ratio": row["amp_ratio"],
                "is_anomalous": int(is_anomalous),
                **window_feats,
            }
        )
    return rows, f0_error


def main():
    config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    labels_df = pd.read_csv(config.LABELS_CSV)

    all_rows = []
    f0_errors = []
    for _, row in labels_df.iterrows():
        rows, f0_error = build_rows_for_file(row, config)
        all_rows.extend(rows)
        f0_errors.append(f0_error)
        print(f"{row['filename']}: {len(rows)} windows, f0_hat error={f0_error:.4f} Hz")

    dataset_df = pd.DataFrame(all_rows)
    dataset_df.to_csv(config.FEATURES_CSV, index=False)

    f0_errors = np.array(f0_errors)
    print(f"\nwrote {config.FEATURES_CSV} ({len(dataset_df)} windows from {len(labels_df)} files)")
    print(f"f0 estimation error: mean={f0_errors.mean():.4f} Hz, max={f0_errors.max():.4f} Hz")
    print(f"class balance: {dataset_df['is_anomalous'].mean():.3f} fraction anomalous")


if __name__ == "__main__":
    main()
