"""
peakfinders/scipy_local_max.py
==============================
Peak finder #1 – scipy morphological local maximum.

Algorithm
---------
1. Optionally uniform-smooth to suppress single hot pixels.
2. Find local maxima via maximum_filter (a pixel is a maximum if it equals
   the maximum within a (min_distance × min_distance) neighbourhood).
3. Apply an absolute threshold:  pixel > threshold_abs  OR
   a relative threshold:  pixel > threshold_rel × image.max()
   whichever is *more* permissive (logical OR).  Set threshold_rel = 0 to
   disable relative thresholding; set threshold_abs = 0 to disable absolute.
4. Label connected components of the surviving maxima map; take the
   intensity-weighted centroid of each component.
5. Sort by intensity (brightest first) and return at most max_peaks results.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter, label, center_of_mass

from peakfinders.base import BasePeakFinder
from data_model import PeakResult


class ScipyLocalMaxFinder(BasePeakFinder):

    NAME = "Scipy Local Max"
    DESCRIPTION = "Morphological local-maximum filter (scipy.ndimage)"

    # ------------------------------------------------------------------
    # Initialise tkinter variables (no widgets yet – Tk may not exist)
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        super().__init__()
        # Variables are created lazily in build_ui so they belong to the
        # correct Tk root (important for multi-root setups).
        self._vars_ready = False

    def _ensure_vars(self) -> None:
        if self._vars_ready:
            return
        self.v_smooth_size   = tk.IntVar(value=3)
        self.v_min_dist      = tk.IntVar(value=10)
        self.v_thresh_rel    = tk.DoubleVar(value=0.05)
        self.v_thresh_abs    = tk.DoubleVar(value=0.0)
        self.v_max_peaks     = tk.IntVar(value=500)
        self._vars_ready = True

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def build_ui(self, parent: tk.Widget) -> None:
        self._ensure_vars()

        # Description subtitle
        ttk.Label(
            parent,
            text=self.DESCRIPTION,
            font=("Helvetica", 8, "italic"),
            foreground="#555",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))

        rows = [
            ("Smooth kernel size (px)",   self.v_smooth_size,  1,    21,   2,    "{:.0f}"),
            ("Min distance (px)",         self.v_min_dist,     1,   100,   1,    "{:.0f}"),
            ("Threshold – relative (×max)", self.v_thresh_rel, 0.0,  1.0, 0.005, "{:.3f}"),
            ("Threshold – absolute (ADU)", self.v_thresh_abs,  0.0, 1e6,  1.0,  "{:.1f}"),
            ("Max peaks",                 self.v_max_peaks,    1,  2000,   1,    "{:.0f}"),
        ]

        for r, (label_text, var, lo, hi, res, fmt) in enumerate(rows, start=1):
            ttk.Label(parent, text=label_text, anchor="w").grid(
                row=r, column=0, sticky="w", pady=3, padx=(0, 6))

            slider = ttk.Scale(parent, variable=var, from_=lo, to=hi,
                               orient=tk.HORIZONTAL, length=200)
            slider.grid(row=r, column=1, sticky="ew", padx=4)

            entry_var = tk.StringVar(value=fmt.format(var.get()))
            entry = ttk.Entry(parent, textvariable=entry_var, width=9)
            entry.grid(row=r, column=2, padx=4)

            # Keep slider ↔ entry in sync.
            # trace_add callbacks receive (name, index, mode) as positional
            # args — use *args so they don't clobber the captured defaults.
            def _s2e(*args, v=var, ev=entry_var, f=fmt):
                ev.set(f.format(v.get()))

            def _e2v(*args, v=var, ev=entry_var, lo=lo, hi=hi):
                try:
                    val = float(ev.get())
                    v.set(max(lo, min(hi, val)))
                except ValueError:
                    pass

            var.trace_add("write", _s2e)
            entry_var.trace_add("write", _e2v)

        parent.columnconfigure(1, weight=1)

    # ------------------------------------------------------------------
    # Algorithm
    # ------------------------------------------------------------------

    def run(self, image: np.ndarray, filepath: Path) -> PeakResult:
        self._ensure_vars()

        smooth_size  = max(1, int(self.v_smooth_size.get()))
        min_dist     = max(1, int(self.v_min_dist.get()))
        thresh_rel   = float(self.v_thresh_rel.get())
        thresh_abs   = float(self.v_thresh_abs.get())
        max_peaks    = max(1, int(self.v_max_peaks.get()))

        params = dict(
            smooth_size=smooth_size,
            min_dist=min_dist,
            thresh_rel=thresh_rel,
            thresh_abs=thresh_abs,
            max_peaks=max_peaks,
        )

        # 1. Smooth
        work = uniform_filter(image, size=smooth_size) if smooth_size > 1 else image.copy()

        # 2. Local maxima
        neighbourhood = maximum_filter(work, size=min_dist)
        local_max = work == neighbourhood

        # 3. Threshold (OR of relative and absolute)
        img_max = work.max()
        mask_rel = work > thresh_rel * img_max if thresh_rel > 0 else np.zeros_like(local_max)
        mask_abs = work > thresh_abs           if thresh_abs > 0 else np.zeros_like(local_max)
        thresh_mask = mask_rel | mask_abs
        if not thresh_mask.any():
            # Fallback: at least above zero
            thresh_mask = work > 0

        candidates = local_max & thresh_mask

        labelled, n_features = label(candidates)
        if n_features == 0:
            return PeakResult.empty(
                source_file=filepath,
                finder_name=self.NAME,
                params=params,
            )

        # 4. Intensity-weighted centroid of each labelled region
        # Use original (unsmoothed) image for intensity measurement
        rows_out, cols_out, intensities_out = [], [], []
        for i in range(1, n_features + 1):
            mask = labelled == i
            # Weighted centroid
            total_w = image[mask].sum()
            if total_w == 0:
                cy, cx = center_of_mass(mask)
            else:
                ys, xs = np.where(mask)
                cy = float((ys * image[mask]).sum() / total_w)
                cx = float((xs * image[mask]).sum() / total_w)
            rows_out.append(cy)
            cols_out.append(cx)
            intensities_out.append(float(image[int(round(cy)), int(round(cx))]))

        rows_arr = np.array(rows_out)
        cols_arr = np.array(cols_out)
        ints_arr = np.array(intensities_out)

        # 5. Sort by intensity, cap at max_peaks
        order = np.argsort(ints_arr)[::-1][:max_peaks]
        rows_arr = rows_arr[order]
        cols_arr = cols_arr[order]
        ints_arr = ints_arr[order]

        return PeakResult.from_arrays(
            rows=rows_arr,
            cols=cols_arr,
            intensities=ints_arr,
            source_file=filepath,
            finder_name=self.NAME,
            params=params,
        )
