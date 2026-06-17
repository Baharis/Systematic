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

from dataclasses import dataclass

import matplotlib as mpl
import numpy as np
from scipy.spatial import KDTree


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PairwiseEval:
    """
    Evaluation result for one diffraction image.

    Attributes
    ----------
    powder_deviation : float
        Mean deviation of nearest peaks from powder distances. Larger = worse.
    angle_entropy: float
        Entropy of kde-azimuthal entropy of pairs' versor angle distribution.
    pairs : ndarray of shape (M, 2), dtype int
        Neighbor graph: pairs[k] = (i, j) means k-th pair connects peaks i & j.
    obs_d_star : ndarray of shape (M,)
        Observed inter-peak distance for each pair, in Å⁻¹.
    powder_deviations : ndarray of shape (M,)
        Deviation off powder |obs - nearest| / |near2 - near1| for each pair.
    colors : ndarray of shape (M, 4)
        RGBA colours for drawing each pair; Green → good, red → bad match.
    """

    powder_deviation:  float
    angle_entropy:     float
    pairs:             np.ndarray       # (M, 2) int
    #obs_d_star:        np.ndarray       # (M,) float
    #powder_deviations: np.ndarray       # (M,) float
    colors:            np.ndarray       # (M, 4) float32

    @classmethod
    def void(cls) -> PairwiseEval:
        return cls(
            powder_deviation = 0.0,
            angle_entropy = 0,
            pairs = np.empty((0, 2), dtype=int),
            #obs_d_star = np.empty(0),
            #powder_deviations = np.empty(0),
            colors = np.empty((0, 4), dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# Colour helper
# ---------------------------------------------------------------------------

def _powder_deviations_to_rgba(deviations: np.ndarray) -> np.ndarray:
    """
    Map fractional errors in [0, 1] to RGBA colours.

        0.0  → green   (0, 0.8, 0, 1)
        0.5  → yellow  (1, 1,   0, 1)
        1.0  → red     (1, 0,   0, 1)

    Uses a piecewise linear ramp so green and red are visually distinct
    even for small errors.

    Parameters
    ----------
    deviations : (M,) array of floats, clipped to [0, 1]

    Returns
    -------
    rgba : (M, 4) float32 array
    """

    e = np.clip(deviations, 0.0, 1.0).astype(np.float32)
    cmap = mpl.colormaps['Spectral_r']
    rgba = cmap(e)
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
    k_neighbours: int = 6,
) -> PairwiseEval:
    """
    Evaluate how well nearest inter-peak distances match the supplied powder
    pattern and how low is the entropy of nearest inter-peak angle distribution.

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
    lines = np.asarray(powder_inv_d, dtype=np.float64)
    lines = lines[lines >= 0]

    if n < 2 or len(powder_inv_d) == 0 or angstrom_per_pixel <= 0 or len(lines) == 0:
        return PairwiseEval.void()

    rc = np.column_stack([rows, cols])
    near_pairs = []
    dd_px, jj = KDTree(rc).query(rc, k=k_neighbours + 1)
    for i, (di_px, ji) in enumerate(zip(dd_px, jj)):
        for d_px, j in zip(di_px, ji):
            pair = (min(i, int(j)), max(i, int(j)))
            if i != j and pair not in near_pairs:
                near_pairs.append(pair)

    near_pairs = np.asarray(near_pairs, dtype=int)
    ii = near_pairs[:, 0]
    jj = near_pairs[:, 1]
    dr2  = rows[ii] - rows[jj]
    dc2  = cols[ii] - cols[jj]
    d_px = np.hypot(dr2, dc2)
    d_astar = d_px * angstrom_per_pixel
    angles = np.mod(np.arctan2(dc2, dr2), np.pi)

    print(lines)
    print(d_astar)
    print(max(d_astar))
    print(max(d_astar))

    if len(d_astar) == 0:
        return PairwiseEval.void()

    # Nearest smaller and larger powder line for every pair
    idx = np.searchsorted(lines, d_astar, side='left')
    upper_line = lines[np.clip(idx, 0, len(lines) - 1)]
    lower_line = lines[np.clip(idx - 1, 0, len(lines) - 1)]
    max_dev = 0.5 * np.max(np.diff(lines))

    abs_dev = np.min(np.abs([upper_line - d_astar, lower_line - d_astar]), axis=0)
    powder_deviation = abs_dev / max_dev
    colors = _powder_deviations_to_rgba(powder_deviation)

    # Calculate circular KDE
    theta, density = circular_kde(angles, kappa=1000)
    kde_entropy = -np.sum(density * np.log(density + 1e-12))
    kde_entropy /= np.log(len(density))
    print(kde_entropy)

    return PairwiseEval(
        powder_deviation=float(np.mean(powder_deviation)),
        angle_entropy=kde_entropy,
        pairs=near_pairs,
        colors=colors,
    )
