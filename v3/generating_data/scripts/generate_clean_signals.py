"""Generate non-stationary baseline signals at different frequencies and
phases, sampled at config.FS for config.DURATION_S seconds, and write each
to its own CSV under raw_good_data/.

Unlike Initial_Run's pure single tone, each "clean" signal here layers a
slow amplitude-modulation drift, a slow frequency-modulation drift, 2nd/3rd
harmonics, and measurement noise onto the fundamental -- a deliberately
harder baseline that breaks the single-tone assumption several downstream
features rely on."""

import numpy as np
import pandas as pd

import config


def make_clean_signal(f0, phase0, amplitude, fs, duration_s, rng):
    n = fs * duration_s
    t = np.arange(n) / fs

    am = 1.0 + config.AMP_DRIFT_DEPTH * np.sin(
        2 * np.pi * config.AMP_DRIFT_FREQ_HZ * t + rng.uniform(0, 2 * np.pi)
    )
    fm_phase_dev = (config.FREQ_DRIFT_FRAC * f0 / config.FREQ_DRIFT_FREQ_HZ) * np.sin(
        2 * np.pi * config.FREQ_DRIFT_FREQ_HZ * t + rng.uniform(0, 2 * np.pi)
    )
    fundamental = amplitude * am * np.sin(2 * np.pi * f0 * t + fm_phase_dev + phase0)

    h2 = config.HARMONIC2_AMP_RATIO * amplitude * np.sin(
        2 * np.pi * 2 * f0 * t + phase0 + rng.uniform(0, 2 * np.pi)
    )
    h3 = config.HARMONIC3_AMP_RATIO * amplitude * np.sin(
        2 * np.pi * 3 * f0 * t + phase0 + rng.uniform(0, 2 * np.pi)
    )

    noise = rng.normal(0.0, config.NOISE_SIGMA, size=n)
    signal = fundamental + h2 + h3 + noise
    return t, signal


def save_clean_csv(t, values, out_path):
    df = pd.DataFrame(
        {
            "time_s": t,
            "sample_idx": np.arange(len(values)),
            "value": values,
        }
    )
    df.to_csv(out_path, index=False)


def main():
    config.RAW_GOOD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.RANDOM_SEED)

    index_rows = []
    base_signal_id = 0
    for f0 in config.BASE_FREQS:
        for phase_idx, phase0 in enumerate(config.BASE_PHASES):
            base_signal_id += 1
            t, values = make_clean_signal(
                f0, phase0, config.AMPLITUDE, config.FS, config.DURATION_S, rng,
            )
            filename = f"clean_f{f0:g}_p{phase_idx}.csv"
            out_path = config.RAW_GOOD_DATA_DIR / filename
            save_clean_csv(t, values, out_path)

            index_rows.append(
                {
                    "base_signal_id": base_signal_id,
                    "filename": filename,
                    "f0": f0,
                    "phase_idx": phase_idx,
                    "phase0": phase0,
                    "amplitude": config.AMPLITUDE,
                }
            )
            print(f"wrote {out_path} (f0={f0} Hz, phase0={phase0:.3f} rad)")

    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(config.BASE_SIGNALS_INDEX_CSV, index=False)
    print(f"wrote {config.BASE_SIGNALS_INDEX_CSV} ({len(index_df)} base signals)")


if __name__ == "__main__":
    main()
