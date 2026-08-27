"""Generate all diagnostic PNGs into plots/: example signals, the
no-spike/mean-preserved injection proof, spectrogram, instantaneous
frequency, feature distributions, feature importances, confusion matrix,
prediction overlays, and the two headline evaluation plots (F1/IoU vs.
amplitude ratio, RF vs. classical-baseline ROC)."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import hilbert, spectrogram
from sklearn.metrics import confusion_matrix, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "generating_data" / "scripts"))
import config
import features as feat


def savefig(fig, name):
    out_path = config.PLOTS_DIR / name
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_clean_examples(index_df):
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    sample = index_df.sample(n=min(3, len(index_df)), random_state=config.RANDOM_SEED)
    for ax, (_, row) in zip(axes, sample.iterrows()):
        df = pd.read_csv(config.RAW_GOOD_DATA_DIR / row["filename"])
        mask = df["time_s"] <= 5
        ax.plot(df["time_s"][mask], df["value"][mask])
        ax.set_title(f"{row['filename']} (f0={row['f0']} Hz, phase0={row['phase0']:.2f} rad)")
        ax.set_ylabel("value")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Clean baseline signals (first 5 s)")
    savefig(fig, "raw_clean_signal_examples.png")


def plot_injection_proof(example_row):
    df = pd.read_csv(config.ANOMALY_DATA_DIR / example_row["filename"])
    t = df["time_s"].to_numpy()
    v = df["value"].to_numpy()
    start_s, end_s = example_row["start_time_s"], example_row["end_time_s"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    axes[0].plot(t, v, linewidth=0.6)
    axes[0].axvspan(start_s, end_s, color="red", alpha=0.2, label="true anomaly window")
    axes[0].set_title(f"{example_row['filename']}: full signal (mean before={example_row['mean_before']:.5f}, "
                       f"mean after={example_row['mean_after']:.5f})")
    axes[0].set_xlabel("time (s)")
    axes[0].legend()

    zoom_lo, zoom_hi = start_s - 0.5, start_s + 0.5
    mask = (t >= zoom_lo) & (t <= zoom_hi)
    axes[1].plot(t[mask], v[mask], marker=".", markersize=3)
    axes[1].axvline(start_s, color="red", linestyle="--", label="anomaly start")
    axes[1].set_title("Zoom on splice boundary (proves no value/derivative discontinuity)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend()

    fig.tight_layout()
    savefig(fig, "anomaly_injection_no_spike_proof.png")
    return df, example_row


def plot_spectrogram(df, example_row):
    v = df["value"].to_numpy()
    f, t_spec, Sxx = spectrogram(v, fs=config.FS, nperseg=2000, noverlap=1500)
    fig, ax = plt.subplots(figsize=(11, 5))
    pcm = ax.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-12), shading="auto")
    ax.set_ylim(0, 20)
    ax.axvspan(example_row["start_time_s"], example_row["end_time_s"], color="red", alpha=0.15)
    ax.set_ylabel("frequency (Hz)")
    ax.set_xlabel("time (s)")
    ax.set_title(f"Spectrogram: {example_row['filename']} (shaded = true anomaly window)")
    fig.colorbar(pcm, ax=ax, label="power (dB)")
    savefig(fig, "anomaly_spectrogram.png")


def plot_instantaneous_frequency(df, example_row):
    v = df["value"].to_numpy()
    t = df["time_s"].to_numpy()
    analytic = hilbert(v)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(inst_phase) / (2 * np.pi) * config.FS
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t[1:], inst_freq, linewidth=0.5)
    ax.axvspan(example_row["start_time_s"], example_row["end_time_s"], color="red", alpha=0.15,
               label="true anomaly window")
    ax.set_ylim(-5, 30)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("instantaneous frequency (Hz)")
    ax.set_title(f"Hilbert instantaneous frequency: {example_row['filename']}")
    ax.legend()
    savefig(fig, "instantaneous_frequency.png")


def plot_feature_boxplots(dataset_df):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, col in zip(axes.flat, feat.FEATURE_COLUMNS):
        data = [
            dataset_df.loc[dataset_df["is_anomalous"] == 0, col],
            dataset_df.loc[dataset_df["is_anomalous"] == 1, col],
        ]
        ax.boxplot(data, tick_labels=["normal", "anomalous"], showfliers=False)
        ax.set_title(col, fontsize=9)
    fig.suptitle("Physics-based feature distributions: normal vs. anomalous windows")
    fig.tight_layout()
    savefig(fig, "feature_distributions.png")


def plot_feature_importances(metrics):
    importances = metrics["feature_importances"]
    names = list(importances.keys())
    values = [importances[n] for n in names]
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([names[i] for i in order], [values[i] for i in order])
    ax.set_xlabel("importance")
    ax.set_title("Random Forest feature importances")
    fig.tight_layout()
    savefig(fig, "feature_importances.png")


def plot_confusion_matrix(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in zip(axes, ["random_forest", "baseline"], ["Random Forest", "LSQ-residual baseline"]):
        cm = np.array(metrics[key]["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_xticks([0, 1], ["normal", "anomalous"])
        ax.set_yticks([0, 1], ["normal", "anomalous"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{title} (F1={metrics[key]['f1']:.3f})")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    savefig(fig, "confusion_matrices.png")


def plot_prediction_overlays(per_file_df, labels_df, n_examples=4):
    examples = per_file_df.dropna(subset=["true_start_s"]).sample(
        n=min(n_examples, len(per_file_df.dropna(subset=["true_start_s"]))),
        random_state=config.RANDOM_SEED,
    )
    fig, axes = plt.subplots(len(examples), 1, figsize=(11, 3 * len(examples)), squeeze=False)
    for ax, (_, row) in zip(axes[:, 0], examples.iterrows()):
        df = pd.read_csv(config.ANOMALY_DATA_DIR / row["filename"])
        ax.plot(df["time_s"], df["value"], linewidth=0.4, color="gray")
        ax.axvspan(row["true_start_s"], row["true_end_s"], color="red", alpha=0.2, label="true window")
        if not np.isnan(row["pred_start_s"]):
            ax.axvspan(row["pred_start_s"], row["pred_end_s"], color="blue", alpha=0.2, label="predicted window")
        ax.set_title(f"{row['filename']} (amp_ratio={row['amp_ratio']:.2f}, IoU={row['iou']:.2f})")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("time (s)")
    fig.tight_layout()
    savefig(fig, "prediction_overlays.png")


def plot_amp_ratio_sweep(sweep_df):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(sweep_df["amp_ratio"], sweep_df["f1_window_level"], "o-", label="F1 (window-level)")
    ax1.plot(sweep_df["amp_ratio"], sweep_df["mean_iou"], "s-", label="mean IoU (reconstructed window)")
    ax1.set_xlabel("anomaly amplitude ratio (A_a / A)")
    ax1.set_ylabel("score")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Detection quality vs. anomaly amplitude ratio (difficulty sweep)")
    ax1.legend()
    savefig(fig, "amplitude_ratio_sweep.png")


def plot_roc_comparison(test_pred_df):
    fig, ax = plt.subplots(figsize=(7, 6))
    for score_col, label in [("y_proba_rf", "Random Forest"), ("lsq_residual_energy", "LSQ-residual baseline")]:
        fpr, tpr, _ = roc_curve(test_pred_df["is_anomalous"], test_pred_df[score_col])
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.7)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC: Random Forest vs. classical baseline")
    ax.legend()
    savefig(fig, "roc_comparison_rf_vs_baseline.png")


def main():
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    index_df = pd.read_csv(config.BASE_SIGNALS_INDEX_CSV)
    labels_df = pd.read_csv(config.LABELS_CSV)
    dataset_df = pd.read_csv(config.FEATURES_CSV)
    with open(config.METRICS_PATH) as f:
        metrics = json.load(f)
    per_file_df = pd.read_csv(config.MODEL_DIR / "eval_per_file.csv")
    sweep_df = pd.read_csv(config.MODEL_DIR / "eval_amp_sweep.csv")
    test_pred_df = pd.read_csv(config.MODEL_DIR / "eval_test_predictions.csv")

    plot_clean_examples(index_df)

    real_anomalies = labels_df[labels_df["start_idx"] >= 0]
    example_row = real_anomalies[np.isclose(real_anomalies["amp_ratio"], 1.0)].iloc[0]
    df, example_row = plot_injection_proof(example_row)
    plot_spectrogram(df, example_row)
    plot_instantaneous_frequency(df, example_row)

    plot_feature_boxplots(dataset_df)
    plot_feature_importances(metrics)
    plot_confusion_matrix(metrics)
    plot_prediction_overlays(per_file_df, labels_df)
    plot_amp_ratio_sweep(sweep_df)
    plot_roc_comparison(test_pred_df)


if __name__ == "__main__":
    main()
