"""Physics-based feature extraction, shared by build_dataset.py (training)
and interactive_viewer.py (inference), so features are computed identically
in both places.

All features are grounded in signal-processing / physics rather than raw
waveform samples: Fourier / Welch spectral analysis, Hilbert-transform
instantaneous frequency, and a least-squares single-tone fit residual (a
matched-filter / energy-detector statistic for a narrowband intrusion).
"""

import sys
from pathlib import Path

import numpy as np
from scipy.signal import hilbert, welch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "generating_data" / "scripts"))
import config

BAND_HALFWIDTH_HZ = 1.0

FEATURE_COLUMNS = [
    "dominant_freq_offset",
    "out_band_power_ratio",
    "inst_freq_std",
    "inst_freq_dev",
    "lsq_residual_energy",
    "zcr_deviation",
    "rms",
    "spectral_entropy",
]


def sliding_windows(n_samples, window_len, stride):
    starts = np.arange(0, n_samples - window_len + 1, stride)
    return [(int(s), int(s + window_len)) for s in starts]


def estimate_f0(values, fs):
    """Dominant frequency of the whole record via Welch periodogram."""
    nperseg = min(len(values), 8192)
    freqs, psd = welch(values, fs=fs, nperseg=nperseg)
    peak_idx = np.argmax(psd[1:]) + 1  # skip DC bin
    return float(freqs[peak_idx])


def _lsq_sinusoid_residual(window_values, t_local, f0_hat):
    """Least-squares fit of a + b*sin(2*pi*f0*t) + c*cos(2*pi*f0*t); return
    mean squared residual. This is the matched-subspace energy-detector
    statistic: near zero for a clean single tone at f0_hat, elevated when
    the window contains any other frequency content."""
    design = np.column_stack(
        [
            np.ones_like(t_local),
            np.sin(2 * np.pi * f0_hat * t_local),
            np.cos(2 * np.pi * f0_hat * t_local),
        ]
    )
    coeffs, *_ = np.linalg.lstsq(design, window_values, rcond=None)
    fitted = design @ coeffs
    residual = window_values - fitted
    return float(np.mean(residual ** 2))


def extract_window_features(window_values, fs, f0_hat):
    n = len(window_values)
    t_local = np.arange(n) / fs
    duration_s = n / fs

    # 1-2: spectral features via Welch PSD of the window
    nperseg = min(n, 1024)
    freqs, psd = welch(window_values, fs=fs, nperseg=nperseg)
    total_power = np.sum(psd) + 1e-12
    dominant_freq = float(freqs[np.argmax(psd)])
    dominant_freq_offset = abs(dominant_freq - f0_hat)

    band_mask = np.abs(freqs - f0_hat) < BAND_HALFWIDTH_HZ
    in_band_power = np.sum(psd[band_mask])
    out_band_power_ratio = float(1.0 - in_band_power / total_power)

    # 3: instantaneous frequency via Hilbert transform
    analytic = hilbert(window_values)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(inst_phase) / (2.0 * np.pi) * fs
    inst_freq_mean = float(np.mean(inst_freq))
    inst_freq_std = float(np.std(inst_freq))
    inst_freq_dev = abs(inst_freq_mean - f0_hat)

    # 4: least-squares single-tone residual (matched-filter / GLRT statistic)
    lsq_residual_energy = _lsq_sinusoid_residual(window_values, t_local, f0_hat)

    # 5: zero-crossing rate deviation from the theoretical rate (2*f0 per s)
    signs = np.sign(window_values)
    signs[signs == 0] = 1
    zero_crossings = np.sum(np.diff(signs) != 0)
    zcr = zero_crossings / duration_s

    expected_zcr = 2.0 * f0_hat
    zcr_deviation = abs(zcr - expected_zcr)

    # 6: RMS amplitude
    rms = float(np.sqrt(np.mean(window_values ** 2)))

    # 7: spectral entropy (normalized to [0, 1])
    p = psd / total_power
    p = p[p > 0]
    entropy = float(-np.sum(p * np.log(p)) / np.log(len(p)))

    return {
        "dominant_freq_offset": dominant_freq_offset,
        "out_band_power_ratio": out_band_power_ratio,
        "inst_freq_std": inst_freq_std,
        "inst_freq_dev": inst_freq_dev,
        "lsq_residual_energy": lsq_residual_energy,
        "zcr_deviation": zcr_deviation,
        "rms": rms,
        "spectral_entropy": entropy,
    }


def label_window(start, end, true_start, true_end, overlap_threshold=None):
    if overlap_threshold is None:
        overlap_threshold = config.OVERLAP_THRESHOLD
    if true_start < 0 or true_end < 0:
        return False
    overlap = max(0, min(end, true_end) - max(start, true_start))
    window_len = end - start
    return (overlap / window_len) >= overlap_threshold
