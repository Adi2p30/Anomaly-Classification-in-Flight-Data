"""Inject one of three anomaly types into each clean signal in
raw_good_data/, write the result to anomaly_data/, and record the
ground-truth anomaly window in anomaly_data/labels.csv.

Three distinct failure modes, each a physically-motivated way a sensor
channel can misbehave:

- freq_swap:   a different-frequency tone rides on top of the signal for a
               while (as in Initial_Run) -- a Tukey-tapered sinusoid, taper-
               weighted DC-corrected so it sums to exactly zero over the
               window: no discontinuity in value or slope at the splice
               boundaries, and the record's global mean is preserved to
               floating-point precision.
- amp_dropout: the sensor's gain drops for a while (attenuation) -- a
               multiplicative Tukey-tapered gain dip, continuous (gain=1)
               at both edges.
- dc_jump:     the sensor's baseline shifts for a while (bias jump) -- a
               raised-cosine shelf, C1-continuous at both edges. Unlike the
               other two, this is *not* zero-mean by construction: a DC
               jump inherently shifts the local (and slightly the global)
               mean, which is the whole point of the anomaly.

Every type still touches zero samples outside its labeled window, which
`inject()` asserts for every file it writes.
"""

import numpy as np
import pandas as pd
from scipy.signal import windows

import config


def build_freq_swap_component(n_total, i0, n_a, f_a, amp_ratio, base_amplitude,
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


def build_amp_dropout_component(values, i0, n_a, depth, tukey_alpha):
    """Multiplicative attenuation: gain(tau) = 1 - depth*tukey(n_a, alpha),
    so gain==1 (no change) at both edges and 1-depth at the flat middle.
    Returns the additive component (out - values) so callers can uniformly
    do out = values + component."""
    n_total = len(values)
    env = windows.tukey(n_a, alpha=tukey_alpha, sym=True)
    gain = 1.0 - depth * env

    component = np.zeros(n_total)
    component[i0:i0 + n_a] = values[i0:i0 + n_a] * (gain - 1.0)
    return component


def build_dc_jump_component(n_total, i0, n_a, offset, ramp_frac):
    """Raised-cosine shelf: 0 -> offset -> 0, C1-continuous at both edges
    (derivative of 0.5*(1-cos(pi*x)) is 0 at x=0 and x=1) -- but NOT
    zero-mean: a DC jump is inherently a mean shift within the window, by
    definition."""
    n_ramp = max(1, int(round(ramp_frac / 2 * n_a)))
    n_ramp = min(n_ramp, n_a // 2)
    rise = 0.5 * (1 - np.cos(np.pi * np.arange(n_ramp) / n_ramp))

    component = np.zeros(n_total)
    component[i0:i0 + n_ramp] = offset * rise
    component[i0 + n_ramp:i0 + n_a - n_ramp] = offset
    component[i0 + n_a - n_ramp:i0 + n_a] = offset * rise[::-1]
    return component


def draw_anomaly_freq(f0, rng, regime):
    choices = config.ANOMALY_FREQ_RATIO_REGIMES[regime]
    lo, hi = choices[rng.integers(0, len(choices))]
    ratio = rng.uniform(lo, hi)
    f_a = f0 * ratio
    return float(np.clip(f_a, config.ANOMALY_FREQ_MIN_HZ, config.ANOMALY_FREQ_MAX_HZ))


def inject(values, fs, amplitude, f0, rng, anomaly_type):
    n_total = len(values)

    duration_s = rng.uniform(*config.ANOMALY_DURATION_RANGE_S)
    n_a = round(duration_s * fs)

    margin = int(config.EDGE_MARGIN_S * fs)
    i0 = int(rng.integers(margin, n_total - margin - n_a))

    meta = {
        "anomaly_type": anomaly_type,
        "anomaly_freq": np.nan,
        "anomaly_phase": np.nan,
        "freq_regime": np.nan,
        "dc_offset": np.nan,
    }

    if anomaly_type == "freq_swap":
        regime = rng.choice(["wide", "narrow"])
        f_a = draw_anomaly_freq(f0, rng, regime)
        amp_ratio = float(rng.choice(config.AMP_RATIO_CHOICES))
        phase_a = float(rng.uniform(0, 2 * np.pi))
        component = build_freq_swap_component(
            n_total, i0, n_a, f_a, amp_ratio, amplitude, phase_a, fs, config.TUKEY_ALPHA
        )
        severity = amp_ratio
        meta.update(anomaly_freq=f_a, anomaly_phase=phase_a, freq_regime=regime)

    elif anomaly_type == "amp_dropout":
        depth = float(rng.choice(config.DROPOUT_DEPTH_CHOICES))
        component = build_amp_dropout_component(values, i0, n_a, depth, config.TUKEY_ALPHA)
        severity = depth

    elif anomaly_type == "dc_jump":
        offset_ratio = float(rng.choice(config.DC_JUMP_OFFSET_RATIO_CHOICES))
        sign = float(rng.choice([-1.0, 1.0]))
        offset = sign * offset_ratio * amplitude
        component = build_dc_jump_component(n_total, i0, n_a, offset, config.TUKEY_ALPHA)
        severity = offset_ratio
        meta.update(dc_offset=offset)

    else:
        raise ValueError(f"unknown anomaly_type: {anomaly_type}")

    out_values = values + component
    is_anomaly = np.zeros(n_total, dtype=int)
    is_anomaly[i0:i0 + n_a] = 1

    mean_before = float(np.mean(values))
    mean_after = float(np.mean(out_values))
    outside = np.ones(n_total, dtype=bool)
    outside[i0:i0 + n_a] = False
    assert np.array_equal(out_values[outside], values[outside]), (
        "samples outside the anomaly window were modified"
    )
    assert abs(component[i0]) < 1e-9 and abs(component[i0 + n_a - 1]) < 1e-9, (
        "anomaly component is not exactly zero (continuous) at splice boundaries"
    )

    if anomaly_type == "freq_swap":
        assert abs(mean_after - mean_before) < 1e-8, (
            f"mean not preserved: {mean_before} -> {mean_after}"
        )
    else:
        # amp_dropout/dc_jump are not zero-mean by construction -- instead
        # assert the bookkeeping identity holds exactly (catches copy/paste
        # bugs in `component` without requiring the mean shift to vanish).
        assert abs((mean_after - mean_before) - component.sum() / n_total) < 1e-12, (
            "mean shift does not match component sum / n_total"
        )

    meta.update({
        "severity": severity,
        "start_idx": i0,
        "end_idx": i0 + n_a,
        "start_time_s": i0 / fs,
        "end_time_s": (i0 + n_a) / fs,
        "mean_before": mean_before,
        "mean_after": mean_after,
    })
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
            anomaly_type = config.ANOMALY_TYPES[k % len(config.ANOMALY_TYPES)]
            out_values, is_anomaly, meta = inject(
                values, config.FS, base_row["amplitude"], base_row["f0"], rng, anomaly_type
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
            print(f"wrote {out_path} ({anomaly_type} {meta['start_time_s']:.1f}s-"
                  f"{meta['end_time_s']:.1f}s, severity={meta['severity']})")

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
                "anomaly_type": "none",
                "severity": 0.0,
                "anomaly_freq": np.nan,
                "anomaly_phase": np.nan,
                "freq_regime": np.nan,
                "dc_offset": np.nan,
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
