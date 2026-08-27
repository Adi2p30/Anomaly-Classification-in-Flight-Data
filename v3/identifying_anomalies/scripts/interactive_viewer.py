"""Standalone interactive matplotlib tool to explore a generated signal.

Usage:
    python interactive_viewer.py [filename] [--model rf|hgb]

filename can be either a clean base signal (e.g. clean_f1_p0.csv, from
raw_good_data/) or an anomaly/control file (e.g. anom_clean_f1_p0_0.csv,
control_clean_f1_p0.csv, from anomaly_data/) -- both are resolved by name
automatically. If no filename is given and the model has been evaluated
(model/eval_per_file.csv exists), you get a numbered menu of the held-out
test-set files -- picked by base_signal_id, so the model has never seen
them -- with their precomputed predicted window and IoU to choose from
(this precomputed window is always the Random Forest's, per
evaluate_and_reconstruct.py). Without an evaluation on disk, it falls back
to the first true-anomaly file in labels.csv. --model selects which
trained model (rf or hgb, default hgb -- the more capable of the two) is
used for on-the-fly prediction on any file not covered by the precomputed
evaluation.

Drag the slider to scrub a zoom window across the full record; use the
radio buttons to switch between the raw waveform, spectrogram, and Hilbert
instantaneous-frequency views. The true anomaly window (red) and the
model's predicted window (blue) are shaded on both the overview and the
zoomed view. For anomaly/control files, the pre-anomaly original signal is
overlaid (black, dashed) next to the anomalous signal (blue, solid) in the
raw view, so it's visually obvious that the two are identical outside the
labeled window and diverge only inside it.

Must be run as a script with an interactive backend and a live display
(not inside a headless environment or `%matplotlib inline`).
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import RadioButtons, Slider
from scipy.signal import hilbert, spectrogram

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "generating_data" / "scripts"))
import config
import features as feat
from reconstruct import reconstruct_window

ZOOM_WIDTH_S = 5.0
OVERVIEW_DECIMATION = 50

MODEL_PATHS = {"rf": config.MODEL_PATH, "hgb": config.HGB_MODEL_PATH}
MODEL_LABELS = {"rf": "Random Forest", "hgb": "HistGradientBoosting"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="?", default=None)
    parser.add_argument("--model", choices=["rf", "hgb"], default="hgb",
                         help="model used for on-the-fly prediction on files with no cached "
                              "evaluation (default: hgb)")
    return parser.parse_args()


def reconstruct_predicted_window(values, fs, f0_hat, model):
    windows = feat.sliding_windows(len(values), config.WINDOW_LEN, config.WINDOW_STRIDE)
    starts = [w[0] for w in windows]
    ends = [w[1] for w in windows]

    # per-file reference scale, needed by dc_offset_dev/rms_ratio -- same
    # two-pass approach as build_dataset.py
    window_rms = [np.sqrt(np.mean(values[s:e] ** 2)) for s, e in windows]
    file_ref = {
        "record_mean": float(np.mean(values)),
        "record_rms": float(np.sqrt(np.mean(values ** 2))),
        "median_window_rms": float(np.median(window_rms)),
    }

    probas = [
        model.predict_proba(
            np.array([[feat.extract_window_features(values[s:e], fs, f0_hat, file_ref)[c]
                       for c in feat.FEATURE_COLUMNS]])
        )[0, 1]
        for s, e in windows
    ]
    return reconstruct_window(
        starts, ends, probas, len(values),
        median_kernel=config.RECONSTRUCT_MEDIAN_KERNEL, hysteresis=config.RECONSTRUCT_HYSTERESIS,
    )


def load_eval_per_file():
    path = config.EVAL_PER_FILE_CSV
    return pd.read_csv(path) if path.exists() else None


def choose_test_file(eval_df):
    eval_df = eval_df.reset_index(drop=True)
    print("held-out test-set files (model was never trained on these base signals):")
    for i, r in eval_df.iterrows():
        true_desc = f"{r['true_start_s']:.1f}s-{r['true_end_s']:.1f}s" if not np.isnan(r["true_start_s"]) else "none (control)"
        print(f"  [{i}] {r['filename']}  anomaly_type={r['anomaly_type']}  severity={r['severity']:.2f}  "
              f"true={true_desc}  iou={r['iou']:.3f}")
    try:
        choice = input(f"pick a number [0-{len(eval_df) - 1}] (Enter for 0): ").strip()
    except EOFError:
        choice = ""
    idx = int(choice) if choice else 0
    idx = max(0, min(idx, len(eval_df) - 1))
    return eval_df.iloc[idx]["filename"]


def resolve_source(requested, labels_df):
    """Return (kind, filename): kind is 'clean' for a raw_good_data file,
    'anomaly' for an anomaly_data file (which may be a true anomaly or a
    noise-free control, per labels.csv)."""
    clean_files = {
        p.name for p in config.RAW_GOOD_DATA_DIR.glob("*.csv")
        if p.name != config.BASE_SIGNALS_INDEX_CSV.name
    }

    if requested is not None:
        if requested in clean_files:
            return "clean", requested
        if requested in labels_df["filename"].values:
            return "anomaly", requested
        print(f"'{requested}' not found; available clean files (raw_good_data/):")
        for f in sorted(clean_files):
            print(f"  {f}")
        print("available anomaly/control files (anomaly_data/):")
        for f in labels_df["filename"]:
            print(f"  {f}")
        sys.exit(1)

    eval_df = load_eval_per_file()
    if eval_df is not None and len(eval_df):
        return "anomaly", choose_test_file(eval_df)

    candidates = labels_df[labels_df["start_idx"] >= 0]
    default = candidates.iloc[0]["filename"]
    print(f"no filename given and no model evaluation on disk, defaulting to {default}")
    return "anomaly", default


def main():
    args = parse_args()
    labels_df = pd.read_csv(config.LABELS_CSV)
    kind, filename = resolve_source(args.filename, labels_df)

    fs = config.FS
    true_start_s = true_end_s = None
    orig_v = None
    row = None
    if kind == "clean":
        df = pd.read_csv(config.RAW_GOOD_DATA_DIR / filename)
    else:
        row = labels_df[labels_df["filename"] == filename].iloc[0]
        df = pd.read_csv(config.ANOMALY_DATA_DIR / filename)
        true_start_s = row["start_idx"] / fs if row["start_idx"] >= 0 else None
        true_end_s = row["end_idx"] / fs if row["end_idx"] >= 0 else None
        orig_df = pd.read_csv(config.RAW_GOOD_DATA_DIR / row["base_file"])
        orig_v = orig_df["value"].to_numpy()

    t = df["time_s"].to_numpy()
    v = df["value"].to_numpy()
    duration = float(t[-1])

    eval_df = load_eval_per_file()
    eval_row = None
    if eval_df is not None:
        match = eval_df[eval_df["filename"] == filename]
        if len(match):
            eval_row = match.iloc[0]

    pred_start_s = pred_end_s = None
    model_path = MODEL_PATHS[args.model]
    if eval_row is not None:
        pred_start_s = None if np.isnan(eval_row["pred_start_s"]) else float(eval_row["pred_start_s"])
        pred_end_s = None if np.isnan(eval_row["pred_end_s"]) else float(eval_row["pred_end_s"])
        pred_desc = f"{pred_start_s}s-{pred_end_s}s" if pred_start_s is not None else "none"
        true_desc = f"{true_start_s}s-{true_end_s}s" if true_start_s is not None else "none"
        print(f"[test-set eval, Random Forest reconstruction] predicted window={pred_desc} "
              f"(true={true_desc}, iou={eval_row['iou']:.3f})")
    elif model_path.exists():
        model = joblib.load(model_path)
        f0_hat = feat.estimate_f0(v, fs)
        pred_start, pred_end = reconstruct_predicted_window(v, fs, f0_hat, model)
        if pred_start is not None:
            pred_start_s, pred_end_s = pred_start / fs, pred_end / fs
        true_desc = f"{true_start_s}s-{true_end_s}s" if true_start_s is not None else "none"
        print(f"[{MODEL_LABELS[args.model]}] estimated f0={f0_hat:.3f} Hz, predicted window="
              f"{pred_start_s}s-{pred_end_s}s (true={true_desc})")
    else:
        print(f"no trained model found at {model_path} -- run train_model.py first for the predicted overlay")

    print("computing spectrogram + instantaneous frequency (once)...")
    f_spec, t_spec, Sxx = spectrogram(v, fs=fs, nperseg=2000, noverlap=1500)
    analytic = hilbert(v)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(inst_phase) / (2 * np.pi) * fs
    t_inst = t[1:]

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 3, 0.3], hspace=0.5)
    ax_overview = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1])
    ax_slider = fig.add_subplot(gs[2])

    if orig_v is not None:
        ax_overview.plot(t[::OVERVIEW_DECIMATION], orig_v[::OVERVIEW_DECIMATION],
                          linewidth=0.7, linestyle="--", color="black", alpha=0.6, label="original")
        ax_overview.plot(t[::OVERVIEW_DECIMATION], v[::OVERVIEW_DECIMATION],
                          linewidth=0.5, color="tab:blue", alpha=0.8, label="anomalous")
    else:
        ax_overview.plot(t[::OVERVIEW_DECIMATION], v[::OVERVIEW_DECIMATION], linewidth=0.5, color="gray")
    if true_start_s is not None:
        ax_overview.axvspan(true_start_s, true_end_s, color="red", alpha=0.2, label="true window")
    if pred_start_s is not None:
        ax_overview.axvspan(pred_start_s, pred_end_s, color="blue", alpha=0.2, label="predicted window")
    ax_overview.set_title(f"{filename} ({kind}) -- overview")
    ax_overview.legend(loc="upper right", fontsize=8)

    zoom_patch = {"p": ax_overview.axvspan(0, ZOOM_WIDTH_S, color="black", alpha=0.15)}

    slider = Slider(ax_slider, "zoom start (s)", 0.0, max(duration - ZOOM_WIDTH_S, 0.01), valinit=0.0)

    radio_ax = fig.add_axes((0.01, 0.35, 0.11, 0.15))
    radio = RadioButtons(radio_ax, ["raw", "spectrogram", "inst. freq"])

    state = {"mode": "raw"}

    def draw_main():
        ax_main.clear()
        start_s = slider.val
        end_s = start_s + ZOOM_WIDTH_S

        if state["mode"] == "raw":
            mask = (t >= start_s) & (t <= end_s)
            if orig_v is not None:
                ax_main.plot(t[mask], orig_v[mask], linewidth=1.2, linestyle="--",
                              color="black", alpha=0.6, label="original")
                ax_main.plot(t[mask], v[mask], linewidth=1.0, color="tab:blue", alpha=0.9, label="anomalous")
            else:
                ax_main.plot(t[mask], v[mask], linewidth=0.8, color="tab:blue")
            ax_main.set_ylabel("value")
        elif state["mode"] == "spectrogram":
            ax_main.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx + 1e-12), shading="auto")
            ax_main.set_ylim(0, 20)
            ax_main.set_ylabel("frequency (Hz)")
        else:
            mask = (t_inst >= start_s) & (t_inst <= end_s)
            ax_main.plot(t_inst[mask], inst_freq[mask], linewidth=0.8)
            ax_main.set_ylim(-5, 30)
            ax_main.set_ylabel("instantaneous frequency (Hz)")

        if true_start_s is not None:
            ax_main.axvspan(true_start_s, true_end_s, color="red", alpha=0.15, label="true window")
        if pred_start_s is not None:
            ax_main.axvspan(pred_start_s, pred_end_s, color="blue", alpha=0.15, label="predicted window")

        ax_main.set_xlim(start_s, end_s)
        ax_main.set_xlabel("time (s)")
        ax_main.set_title(f"mode: {state['mode']}")
        ax_main.legend(loc="upper right", fontsize=8)
        fig.canvas.draw_idle()

    def on_slider_change(val):
        zoom_patch["p"].remove()
        zoom_patch["p"] = ax_overview.axvspan(val, val + ZOOM_WIDTH_S, color="black", alpha=0.15)
        draw_main()

    def on_mode_change(label):
        state["mode"] = {"raw": "raw", "spectrogram": "spectrogram", "inst. freq": "inst_freq"}[label]
        draw_main()

    slider.on_changed(on_slider_change)
    radio.on_clicked(on_mode_change)

    draw_main()
    plt.show()


if __name__ == "__main__":
    main()
