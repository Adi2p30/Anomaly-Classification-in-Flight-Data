"""Physics-based feature extraction, shared by build_dataset.py (training)
and interactive_viewer.py (inference), so features are computed identically
in both places.

All features are grounded in signal-processing / physics rather than raw
waveform samples: Fourier / Welch spectral analysis, Hilbert-transform
instantaneous frequency, a least-squares single-tone fit residual (a
matched-filter / energy-detector statistic for a narrowband intrusion), and
-- new in v3 -- vibration/fault-diagnosis-inspired statistics (spectral
kurtosis, Hilbert-envelope spectrum analysis, crest factor, time-domain
kurtosis, autocorrelation-based periodicity, and two features computed
relative to each file's own reference scale) added to catch anomaly types
(amplitude dropout, DC offset jump) that the original 8 features -- tuned
for a "wrong frequency" intrusion -- are structurally blind to.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.signal import detrend, hilbert, stft, welch
from scipy.stats import kurtosis

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
    "spectral_kurtosis",
    "envelope_pnr",
    "crest_factor",
    "kurtosis_time",
    "periodicity_strength",
    "dc_offset_dev",
    "rms_ratio",
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


def extract_window_features(window_values, fs, f0_hat, file_ref):
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

    # 8: spectral kurtosis (Antoni) via STFT -- per-frequency-bin kurtosis
    # across time frames flags bands with impulsive/transient energy, which
    # dc_jump/amp_dropout edges and short freq_swap bursts produce but a
    # stationary tone does not. 90th percentile (not max) because each bin
    # only has ~30 STFT frames here, making max() a noisy statistic.
    _, _, Zxx = stft(window_values, fs=fs, nperseg=128, noverlap=64)
    sk_per_freq = kurtosis(np.abs(Zxx) ** 2, axis=1, fisher=True, bias=False)
    spectral_kurtosis = float(np.nanpercentile(sk_per_freq, 90))

    # 9: Hilbert-envelope spectrum peak-to-noise ratio -- classic
    # bearing-fault envelope analysis: detrend the envelope first (removes
    # the AM-drift ramp, which would otherwise dominate the lowest PSD
    # bins), then look for a strong periodic component in the envelope
    # itself (amplitude modulation / beating between f0 and an intruder).
    envelope_detrended = detrend(np.abs(analytic), type="linear")
    freqs_env, psd_env = welch(envelope_detrended, fs=fs, nperseg=min(n, 1024))
    envelope_pnr = float(np.max(psd_env[1:]) / (np.median(psd_env[1:]) + 1e-12))

    # 10: crest factor -- peak/RMS; catches transient peaks from
    # dc_jump/dropout edges
    crest_factor = float(np.max(np.abs(window_values)) / (rms + 1e-12))

    # 11: time-domain kurtosis (impulsiveness) -- a pure sine has kurtosis
    # ~ -1.5 (Fisher); sharp steps/transients raise it
    kurtosis_time = float(kurtosis(window_values, fisher=True, bias=False))

    # 12: one-period-lag Pearson autocorrelation (not raw np.correlate --
    # that has a triangular-window bias that would make it a proxy for f0
    # itself rather than periodicity). ~1 for a clean single tone, degraded
    # by any of the three anomaly types.
    lag = max(1, min(int(round(fs / f0_hat)), n - 2))
    seg_a, seg_b = window_values[:n - lag], window_values[lag:]
    periodicity_strength = float(np.corrcoef(seg_a, seg_b)[0, 1]) if len(seg_a) > 1 else 0.0
    if np.isnan(periodicity_strength):
        periodicity_strength = 0.0

    # 13: DC-offset deviation relative to this file's own reference
    # mean/scale -- the feature that actually makes dc_jump detectable:
    # lsq_residual_energy's a+b*sin+c*cos fit absorbs a constant offset
    # entirely into `a`, so the fit residual stays ~0 on the plateau
    # interior.
    dc_offset_dev = float(
        abs(np.mean(window_values) - file_ref["record_mean"]) / (file_ref["record_rms"] + 1e-12)
    )

    # 14: RMS ratio relative to this file's own median window RMS -- what
    # makes amp_dropout detectable across a 7-frequency, drifting-amplitude
    # grid, where raw `rms` alone conflates anomaly severity with which
    # base signal it is.
    rms_ratio = float(rms / (file_ref["median_window_rms"] + 1e-12))

    return {
        "dominant_freq_offset": dominant_freq_offset,
        "out_band_power_ratio": out_band_power_ratio,
        "inst_freq_std": inst_freq_std,
        "inst_freq_dev": inst_freq_dev,
        "lsq_residual_energy": lsq_residual_energy,
        "zcr_deviation": zcr_deviation,
        "rms": rms,
        "spectral_entropy": entropy,
        "spectral_kurtosis": spectral_kurtosis,
        "envelope_pnr": envelope_pnr,
        "crest_factor": crest_factor,
        "kurtosis_time": kurtosis_time,
        "periodicity_strength": periodicity_strength,
        "dc_offset_dev": dc_offset_dev,
        "rms_ratio": rms_ratio,
    }


def label_window(start, end, true_start, true_end, overlap_threshold=None):
    if overlap_threshold is None:
        overlap_threshold = config.OVERLAP_THRESHOLD
    if true_start < 0 or true_end < 0:
        return False
    overlap = max(0, min(end, true_end) - max(start, true_start))
    window_len = end - start
    return (overlap / window_len) >= overlap_threshold
