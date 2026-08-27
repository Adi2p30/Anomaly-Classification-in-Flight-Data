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
"""

import numpy as np


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


def reconstruct_window(window_starts, window_ends, scores, n_total, threshold=0.5):
    """Interpolate per-window scores (assigned to window centers) to
    per-sample resolution and return the largest contiguous run above
    threshold as (start_idx, end_idx), or (None, None) if none exceed it."""
    centers = (np.asarray(window_starts) + np.asarray(window_ends)) / 2.0
    order = np.argsort(centers)
    centers = centers[order]
    scores = np.asarray(scores)[order]

    sample_idx = np.arange(n_total)
    score_curve = np.interp(sample_idx, centers, scores)
    mask = score_curve >= threshold
    return largest_run(mask)
