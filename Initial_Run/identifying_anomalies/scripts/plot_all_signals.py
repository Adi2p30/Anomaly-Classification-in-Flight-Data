"""Generate one PNG per raw clean signal (plots/clean_signals/) and one PNG
per anomaly-injected signal with the true anomaly window shaded
(plots/anomaly_signals/)."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "generating_data" / "scripts"))
import config

CLEAN_PLOTS_DIR = config.PLOTS_DIR / "clean_signals"
ANOMALY_PLOTS_DIR = config.PLOTS_DIR / "anomaly_signals"


def plot_clean_signal(row):
    df = pd.read_csv(config.RAW_GOOD_DATA_DIR / row["filename"])
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["time_s"], df["value"], linewidth=0.4)
    ax.set_title(f"{row['filename']} (f0={row['f0']} Hz, phase0={row['phase0']:.2f} rad)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("value")
    fig.tight_layout()
    out_path = CLEAN_PLOTS_DIR / f"{Path(row['filename']).stem}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_anomaly_signal(row):
    df = pd.read_csv(config.ANOMALY_DATA_DIR / row["filename"])
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["time_s"], df["value"], linewidth=0.4)
    if row["start_idx"] >= 0:
        ax.axvspan(row["start_time_s"], row["end_time_s"], color="red", alpha=0.2, label="true anomaly window")
        ax.legend(loc="upper right", fontsize=8)
        title = (f"{row['filename']} (base={row['base_file']}, amp_ratio={row['amp_ratio']:.2f}, "
                 f"anomaly_freq={row['anomaly_freq']:.2f} Hz)")
    else:
        title = f"{row['filename']} (control, no anomaly)"
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("value")
    fig.tight_layout()
    out_path = ANOMALY_PLOTS_DIR / f"{Path(row['filename']).stem}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def main():
    CLEAN_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    ANOMALY_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    index_df = pd.read_csv(config.BASE_SIGNALS_INDEX_CSV)
    for _, row in index_df.iterrows():
        out_path = plot_clean_signal(row)
        print(f"wrote {out_path}")

    labels_df = pd.read_csv(config.LABELS_CSV)
    for _, row in labels_df.iterrows():
        out_path = plot_anomaly_signal(row)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
