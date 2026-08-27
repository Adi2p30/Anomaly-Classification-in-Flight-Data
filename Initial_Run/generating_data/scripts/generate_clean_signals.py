"""Generate clean sine-wave baseline signals at different frequencies and
phases, sampled at config.FS for config.DURATION_S seconds, and write each
to its own CSV under raw_good_data/."""

import numpy as np
import pandas as pd

import config


def make_clean_signal(f0, phase0, amplitude, fs, duration_s):
    n = fs * duration_s
    t = np.arange(n) / fs
    signal = amplitude * np.sin(2 * np.pi * f0 * t + phase0)
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

    index_rows = []
    base_signal_id = 0
    for f0 in config.BASE_FREQS:
        for phase_idx, phase0 in enumerate(config.BASE_PHASES):
            base_signal_id += 1
            t, values = make_clean_signal(
                f0, phase0, config.AMPLITUDE, config.FS, config.DURATION_S,
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
