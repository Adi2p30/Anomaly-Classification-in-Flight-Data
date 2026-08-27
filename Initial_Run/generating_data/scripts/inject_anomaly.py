"""Inject a smooth, mean-preserving 'wrong frequency' anomaly into each clean
signal in raw_good_data/, write the result to anomaly_data/, and record the
ground-truth anomaly window in anomaly_data/labels.csv.

Injection method (see docstring on build_anomaly_component for the math):
a Tukey-tapered sinusoid at a different frequency is added over a random
window, with a taper-weighted DC correction so the added component sums to
exactly zero over the window -- this guarantees (a) no discontinuity in
value or slope at the splice boundaries (no spike) and (b) the global mean
of the full record is preserved to floating-point precision, without
touching a single sample outside the labeled window.
"""

import numpy as np
import pandas as pd
from scipy.signal import windows

import config


def build_anomaly_component(n_total, i0, n_a, f_a, amp_ratio, base_amplitude,
                             phase_a, fs, tukey_alpha):
    """Return a length-n_total array, zero everywhere except [i0, i0+n_a),
    where it holds a Tukey-tapered, zero-mean sinusoid at f_a."""
    tau = np.arange(n_a) / fs
    env = windows.tukey(n_a, alpha=tukey_alpha, sym=True)
    raw_a = amp_ratio * base_amplitude * np.sin(2 * np.pi * f_a * tau + phase_a)

    k = np.sum(env * raw_a) / np.sum(env)
    c = env * raw_a - k * env

    component = np.zeros(n_total)
    component[i0:i0 + n_a] = c
    return component


def draw_anomaly_freq(f0, rng):
    lo, hi = config.ANOMALY_FREQ_RATIO_CHOICES[
        rng.integers(0, len(config.ANOMALY_FREQ_RATIO_CHOICES))
    ]
    ratio = rng.uniform(lo, hi)
    f_a = f0 * ratio
    return float(np.clip(f_a, config.ANOMALY_FREQ_MIN_HZ, config.ANOMALY_FREQ_MAX_HZ))


def inject(values, fs, amplitude, f0, rng):
    n_total = len(values)

    duration_s = rng.uniform(*config.ANOMALY_DURATION_RANGE_S)
    n_a = round(duration_s * fs)

    margin = int(config.EDGE_MARGIN_S * fs)
    i0 = int(rng.integers(margin, n_total - margin - n_a))

    f_a = draw_anomaly_freq(f0, rng)
    amp_ratio = float(rng.choice(config.AMP_RATIO_CHOICES))
    phase_a = float(rng.uniform(0, 2 * np.pi))

    component = build_anomaly_component(
        n_total, i0, n_a, f_a, amp_ratio, amplitude, phase_a, fs, config.TUKEY_ALPHA
    )

    out_values = values + component
    is_anomaly = np.zeros(n_total, dtype=int)
    is_anomaly[i0:i0 + n_a] = 1

    mean_before = float(np.mean(values))
    mean_after = float(np.mean(out_values))
    assert abs(mean_after - mean_before) < 1e-8, (
        f"mean not preserved: {mean_before} -> {mean_after}"
    )
    outside = np.ones(n_total, dtype=bool)
    outside[i0:i0 + n_a] = False
    assert np.array_equal(out_values[outside], values[outside]), (
        "samples outside the anomaly window were modified"
    )
    assert abs(component[i0]) < 1e-9 and abs(component[i0 + n_a - 1]) < 1e-9, (
        "anomaly component is not exactly zero at splice boundaries"
    )

    meta = {
        "amp_ratio": amp_ratio,
        "anomaly_freq": f_a,
        "anomaly_phase": phase_a,
        "start_idx": i0,
        "end_idx": i0 + n_a,
        "start_time_s": i0 / fs,
        "end_time_s": (i0 + n_a) / fs,
        "mean_before": mean_before,
        "mean_after": mean_after,
    }
    return out_values, is_anomaly, meta


def save_anomaly_csv(t, values, is_anomaly, out_path):
    df = pd.DataFrame(
        {
            "time_s": t,
            "sample_idx": np.arange(len(values)),
            "value": values,
            "is_anomaly": is_anomaly,
        }
    )
    df.to_csv(out_path, index=False)


def main():
    config.ANOMALY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.RANDOM_SEED)

    index_df = pd.read_csv(config.BASE_SIGNALS_INDEX_CSV)
    label_rows = []

    for _, base_row in index_df.iterrows():
        base_path = config.RAW_GOOD_DATA_DIR / base_row["filename"]
        clean_df = pd.read_csv(base_path)
        t = clean_df["time_s"].to_numpy()
        values = clean_df["value"].to_numpy()

        for k in range(config.INJECTIONS_PER_BASE):
            out_values, is_anomaly, meta = inject(
                values, config.FS, base_row["amplitude"], base_row["f0"], rng
            )
            out_filename = f"anom_{base_row['filename'][:-4]}_{k}.csv"
            out_path = config.ANOMALY_DATA_DIR / out_filename
            save_anomaly_csv(t, out_values, is_anomaly, out_path)

            label_rows.append(
                {
                    "filename": out_filename,
                    "base_file": base_row["filename"],
                    "base_signal_id": base_row["base_signal_id"],
                    "f0": base_row["f0"],
                    "phase0": base_row["phase0"],
                    **meta,
                }
            )
            print(f"wrote {out_path} (anomaly {meta['start_time_s']:.1f}s-"
                  f"{meta['end_time_s']:.1f}s @ {meta['anomaly_freq']:.2f} Hz, "
                  f"amp_ratio={meta['amp_ratio']})")

    # anomaly-free control files: copy a few base signals through unmodified
    control_rows = index_df.sample(
        n=min(config.N_CONTROL_FILES, len(index_df)),
        random_state=config.RANDOM_SEED,
    )
    for _, base_row in control_rows.iterrows():
        base_path = config.RAW_GOOD_DATA_DIR / base_row["filename"]
        clean_df = pd.read_csv(base_path)
        t = clean_df["time_s"].to_numpy()
        values = clean_df["value"].to_numpy()
        is_anomaly = np.zeros(len(values), dtype=int)

        out_filename = f"control_{base_row['filename'][:-4]}.csv"
        out_path = config.ANOMALY_DATA_DIR / out_filename
        save_anomaly_csv(t, values, is_anomaly, out_path)

        mean_val = float(np.mean(values))
        label_rows.append(
            {
                "filename": out_filename,
                "base_file": base_row["filename"],
                "base_signal_id": base_row["base_signal_id"],
                "f0": base_row["f0"],
                "phase0": base_row["phase0"],
                "amp_ratio": 0.0,
                "anomaly_freq": np.nan,
                "anomaly_phase": np.nan,
                "start_idx": -1,
                "end_idx": -1,
                "start_time_s": -1.0,
                "end_time_s": -1.0,
                "mean_before": mean_val,
                "mean_after": mean_val,
            }
        )
        print(f"wrote {out_path} (anomaly-free control)")

    labels_df = pd.DataFrame(label_rows)
    labels_df.to_csv(config.LABELS_CSV, index=False)
    print(f"wrote {config.LABELS_CSV} ({len(labels_df)} files)")


if __name__ == "__main__":
    main()
