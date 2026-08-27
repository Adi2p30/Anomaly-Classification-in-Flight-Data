"""Turn per-window anomaly scores into a precise sample-level predicted
interval. Shared by evaluate_and_reconstruct.py and interactive_viewer.py so
both use identical reconstruction logic.

Naively taking the union of windows with a binary positive prediction
quantizes the predicted boundary to the nearest window_stride (0.2 s here).
Instead we assign each window's anomaly *probability* to its center time,
linearly interpolate that score to every sample, and take the 0.5-crossing
of the interpolated curve as the boundary. This recovers much finer
boundary precision than the window grid itself -- limited mainly by how
sharply the classifier's output actually transitions near the (inherently
smooth, tapered) true edge, not by the sliding-window stride.

v3 adds two optional, off-by-default smoothing steps for the noisier
pipeline: a median filter over the window-level scores before
interpolation, and a hysteresis (two-threshold) mask instead of a single
crossing -- both meant to suppress spurious control-file detections now
that measurement noise is present, at some cost to boundary sharpness.
"""

import numpy as np
from scipy.signal import medfilt


def largest_run(mask):
    """Return (start, end) of the longest contiguous run of True values in
    mask, or (None, None) if mask is all False."""
    if not mask.any():
        return None, None
    padded = np.concatenate([[False], mask, [False]])
    edges = np.diff(padded.astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    best = int(np.argmax(ends - starts))
    return int(starts[best]), int(ends[best])


def smooth_scores(scores, kernel_size):
    """Median-filter the window-level score sequence. No-op if kernel_size
    is falsy/<=1 or larger than the number of scores."""
    if not kernel_size or kernel_size <= 1 or len(scores) < kernel_size:
        return scores
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return medfilt(scores, kernel_size=k)


def hysteresis_mask(curve, enter_thr, exit_thr):
    """Two-threshold (Schmitt-trigger) mask: once the curve rises above
    enter_thr the mask stays True until the curve drops below exit_thr --
    more robust to noise dithering around a single threshold than a plain
    curve >= threshold comparison."""
    mask = np.zeros(len(curve), dtype=bool)
    state = False
    for i, v in enumerate(curve):
        if not state and v >= enter_thr:
            state = True
        elif state and v < exit_thr:
            state = False
        mask[i] = state
    return mask


def reconstruct_window(window_starts, window_ends, scores, n_total,
                        threshold=0.5, median_kernel=None, hysteresis=None):
    """Interpolate per-window scores (assigned to window centers) to
    per-sample resolution and return the largest contiguous run above
    threshold as (start_idx, end_idx), or (None, None) if none exceed it.
    If median_kernel is given, scores are median-filtered first. If
    hysteresis=(enter_thr, exit_thr) is given, it's used instead of a
    plain single-threshold crossing."""
    centers = (np.asarray(window_starts) + np.asarray(window_ends)) / 2.0
    order = np.argsort(centers)
    centers = centers[order]
    scores = smooth_scores(np.asarray(scores)[order], median_kernel)

    sample_idx = np.arange(n_total)
    score_curve = np.interp(sample_idx, centers, scores)
    mask = hysteresis_mask(score_curve, *hysteresis) if hysteresis else (score_curve >= threshold)
    return largest_run(mask)
