"""
peak_eval.py
============
Powder-pattern matching for diffraction images.

Given a set of peak positions (in pixels) and a powder pattern (set of
allowed reciprocal-space spacings in Å⁻¹), this module asks:

    "Are the inter-peak distances consistent with a single-crystal zone axis
     of the supplied unit cell?"

Algorithm
---------
1.  Compute all pairwise distances between detected peaks, converting from
    pixels to Å⁻¹ using the supplied calibration factor.

2.  For each observed distance, find the nearest allowed powder-line spacing.
    Compute the fractional deviation:

        rel_error_i = |d_obs_i  -  d_nearest_i| / d_nearest_i

3.  The match score is:

        score = median( 1 - clip(rel_error, 0, 1) )   ∈ [0, 1]

    Interpretation
    ~~~~~~~~~~~~~~
    1.0  Perfect match (all inter-peak distances land on powder lines)
    ~0.9 Good single-crystal data with small calibration error
    ~0.5 Mixed / polycrystalline or poor calibration
    ~0.0 No agreement with the supplied cell

    The median rather than the mean is used so that a minority of outlier
    distances (e.g. spurious peaks, satellite reflections) does not
    disproportionately penalise a good pattern.

4.  Each pair is assigned a colour interpolated from green (rel_error = 0)
    through yellow to red (rel_error = 1), for drawing on the image.

Public API
----------
    PairwiseEval        – result container for one image
    evaluate_image()    – main entry point
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PairwiseEval:
    """
    Evaluation result for one diffraction image.

    Attributes
    ----------
    score : float
        Scalar match quality in [0, 1].  Higher is better.
    pairs : ndarray of shape (M, 2), dtype int
        Indices into the peak array: pairs[k] = (i, j) means the k-th pair
        connects peak i and peak j.
    obs_inv_d : ndarray of shape (M,)
        Observed inter-peak distance for each pair, in Å⁻¹.
    nearest_inv_d : ndarray of shape (M,)
        Nearest allowed powder-line spacing for each pair, in Å⁻¹.
    rel_errors : ndarray of shape (M,)
        Fractional deviation |obs - nearest| / nearest for each pair.
    colors : ndarray of shape (M, 4)
        RGBA colours (float32, values in [0, 1]) for drawing each pair.
        Green → good match, yellow → moderate, red → bad match.
    n_peaks : int
        Number of peaks in the image.
    """

    score:         float
    pairs:         np.ndarray       # (M, 2) int
    obs_inv_d:     np.ndarray       # (M,) float
    nearest_inv_d: np.ndarray       # (M,) float
    rel_errors:    np.ndarray       # (M,) float
    colors:        np.ndarray       # (M, 4) float32
    n_peaks:       int


# ---------------------------------------------------------------------------
# Colour helper
# ---------------------------------------------------------------------------

def _error_to_rgba(rel_errors: np.ndarray) -> np.ndarray:
    """
    Map fractional errors in [0, 1] to RGBA colours.

        0.0  → green   (0, 0.8, 0, 1)
        0.5  → yellow  (1, 1,   0, 1)
        1.0  → red     (1, 0,   0, 1)

    Uses a piecewise linear ramp so green and red are visually distinct
    even for small errors.

    Parameters
    ----------
    rel_errors : (M,) array of floats, clipped to [0, 1]

    Returns
    -------
    rgba : (M, 4) float32 array
    """
    e = np.clip(rel_errors, 0.0, 1.0).astype(np.float32)
    rgba = np.zeros((len(e), 4), dtype=np.float32)
    rgba[:, 3] = 1.0                        # alpha always 1

    # Green → yellow for e in [0, 0.5]: R rises 0→1, G stays 0.8→1
    low = e <= 0.5
    t = e[low] * 2.0                        # 0 → 1 over [0, 0.5]
    rgba[low, 0] = t                        # R: 0 → 1
    rgba[low, 1] = 0.8 + 0.2 * t           # G: 0.8 → 1.0
    rgba[low, 2] = 0.0

    # Yellow → red for e in [0.5, 1.0]: R stays 1, G falls 1→0
    high = ~low
    t = (e[high] - 0.5) * 2.0              # 0 → 1 over [0.5, 1.0]
    rgba[high, 0] = 1.0
    rgba[high, 1] = 1.0 - t                # G: 1 → 0
    rgba[high, 2] = 0.0

    return rgba


# ---------------------------------------------------------------------------
# Neighbourhood filter
# ---------------------------------------------------------------------------

def _nearest_neighbour_pairs(
    rows: np.ndarray,
    cols: np.ndarray,
    k_neighbours: int,
) -> np.ndarray:
    """
    Return (M, 2) array of index pairs (i, j) with i < j such that j is
    among the k nearest neighbours of i (or vice-versa).

    Drawing all N*(N-1)/2 pairs clutters the image for N > ~10.  Restricting
    to k-nearest neighbours preserves the local connectivity that matters for
    single-crystal identification while keeping the drawing tractable.
    """
    n = len(rows)
    if n == 0:
        return np.empty((0, 2), dtype=int)

    pts = np.column_stack([rows, cols])         # (N, 2)
    k = min(k_neighbours, n - 1)

    pair_set: set[tuple[int, int]] = set()
    for i in range(n):
        diffs = pts - pts[i]                    # (N, 2)
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        dists[i] = np.inf                       # exclude self
        nn_idx = np.argsort(dists)[:k]
        for j in nn_idx:
            pair_set.add((min(i, int(j)), max(i, int(j))))

    pairs = np.array(sorted(pair_set), dtype=int)
    return pairs


import numpy as np
from scipy.stats import vonmises

def circular_kde(
    angles: np.ndarray,
    n_grid: int = 360,
    kappa: float = 10.0,
):
    """
    KDE for angles in [0, pi).

    Parameters
    ----------
    angles
        Angles in radians in [0, pi).
    n_grid
        Number of evaluation points.
    kappa
        Concentration parameter.
        Larger = narrower peaks.

    Returns
    -------
    theta : (n_grid,)
        Evaluation points.
    density : (n_grid,)
        Normalized density.
    """
    theta = np.linspace(0, np.pi, n_grid, endpoint=False)

    density = np.zeros_like(theta)

    # Double angles to handle pi-periodicity
    angles2 = 2 * angles
    theta2 = 2 * theta

    for a in angles2:
        density += vonmises.pdf(theta2, kappa, loc=a)

    density /= density.sum()

    return theta, density

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate_image(
    rows: np.ndarray,
    cols: np.ndarray,
    powder_inv_d: np.ndarray,
    angstrom_per_pixel: float,
    k_neighbours: int = 4,
) -> PairwiseEval:
    """
    Evaluate how well inter-peak distances match the supplied powder pattern.

    Parameters
    ----------
    rows, cols : 1-D float arrays of length N
        Peak positions in pixel coordinates.
    powder_inv_d : 1-D float array
        Allowed reciprocal-space spacings in Å⁻¹, as returned by
        ``powder.powder_lines()``.  Must be sorted ascending and positive.
    angstrom_per_pixel : float
        Calibration factor: Å⁻¹ per pixel.
    k_neighbours : int
        Number of nearest neighbours to draw lines for (default 6).
        All pairwise distances are used for the score regardless.

    Returns
    -------
    PairwiseEval
        Score, per-pair errors, colours, and the pair index array.
        If fewer than 2 peaks are present, returns a zero-score result
        with empty arrays.
    """
    n = len(rows)

    if n < 2 or len(powder_inv_d) == 0 or angstrom_per_pixel <= 0:
        return PairwiseEval(
            score=0.0,
            pairs=np.empty((0, 2), dtype=int),
            obs_inv_d=np.empty(0),
            nearest_inv_d=np.empty(0),
            rel_errors=np.empty(0),
            colors=np.empty((0, 4), dtype=np.float32),
            n_peaks=n,
        )

    lines = np.asarray(powder_inv_d, dtype=np.float64)
    lines = lines[lines > 0]
    if len(lines) == 0:
        return PairwiseEval(
            score=0.0,
            pairs=np.empty((0, 2), dtype=int),
            obs_inv_d=np.empty(0),
            nearest_inv_d=np.empty(0),
            rel_errors=np.empty(0),
            colors=np.empty((0, 4), dtype=np.float32),
            n_peaks=n,
        )

    # ── All pairwise distances (for the score) ────────────────────────────
    # Vectorised: build upper-triangle index arrays
    ii, jj = np.triu_indices(n, k=1)           # all unique pairs
    dr = rows[ii] - rows[jj]
    dc = cols[ii] - cols[jj]
    d_px = np.hypot(dr, dc)

    # Convert to Å⁻¹
    obs = d_px * angstrom_per_pixel             # (M,) Å⁻¹
    valid = obs > 1e-9
    obs_valid   = obs[valid]
    ii_valid    = ii[valid]
    jj_valid    = jj[valid]

    if len(obs_valid) == 0:
        return PairwiseEval(
            score=0.0,
            pairs=np.empty((0, 2), dtype=int),
            obs_inv_d=np.empty(0),
            nearest_inv_d=np.empty(0),
            rel_errors=np.empty(0),
            colors=np.empty((0, 4), dtype=np.float32),
            n_peaks=n,
        )

    # Nearest powder line for every pair — shape (M, L) → argmin over L
    diff = np.abs(obs_valid[:, None] - lines[None, :])   # (M, L)
    nearest_idx  = np.argmin(diff, axis=1)               # (M,)
    nearest_val  = lines[nearest_idx]                    # (M,)
    rel_errors_all = np.abs(obs_valid - nearest_val) / nearest_val

    # ── Score: median match quality over ALL pairs ───────────────────────
    score = float(np.median(1.0 - np.clip(rel_errors_all, 0.0, 1.0)))

    # ── Nearest-neighbour pairs (for drawing) ────────────────────────────
    draw_pairs = _nearest_neighbour_pairs(rows, cols, k_neighbours)  # (K, 2)

    if len(draw_pairs) == 0:
        return PairwiseEval(
            score=score,
            pairs=draw_pairs,
            obs_inv_d=np.empty(0),
            nearest_inv_d=np.empty(0),
            rel_errors=np.empty(0),
            colors=np.empty((0, 4), dtype=np.float32),
            n_peaks=n,
        )

    # Re-compute per-pair quantities for the drawing subset
    di_draw = draw_pairs[:, 0]
    dj_draw = draw_pairs[:, 1]
    dr2  = rows[di_draw] - rows[dj_draw]
    dc2  = cols[di_draw] - cols[dj_draw]
    d_px2 = np.hypot(dr2, dc2)
    obs_draw = d_px2 * angstrom_per_pixel

    valid2 = obs_draw > 1e-9
    obs_draw_v     = obs_draw[valid2]
    draw_pairs_v   = draw_pairs[valid2]

    nearest_idx2  = np.argmin(np.abs(obs_draw_v[:, None] - lines[None, :]), axis=1)
    nearest_val2  = lines[nearest_idx2]
    rel_err2      = np.abs(obs_draw_v - nearest_val2) / nearest_val2

    colors = _error_to_rgba(rel_err2)

    # Calculate rotational entropy of nearest (draw) peaks
    from scipy.stats import entropy
    angles = np.mod(np.arctan2(dc2, dr2), np.pi)
    print(angles)
    print(len(angles))
    angle_hist, _ = np.histogram(angles, bins=len(angles), range=(0, np.pi))
    print(angle_hist)
    print(len(angle_hist))
    angle_entropy = entropy(angle_hist)
    print(angle_entropy)

    from matplotlib import pyplot as plt

    # Calculate circular KDE
    theta, density = circular_kde(angles, kappa=1000)
    plt.plot(density)
    plt.show()
    kde_entropy = -np.sum(density * np.log(density + 1e-12))
    kde_entropy /= np.log(len(density))
    print(kde_entropy)

    return PairwiseEval(
        score=score,
        pairs=draw_pairs_v,
        obs_inv_d=obs_draw_v,
        nearest_inv_d=nearest_val2,
        rel_errors=rel_err2,
        colors=colors,
        n_peaks=n,
    )
