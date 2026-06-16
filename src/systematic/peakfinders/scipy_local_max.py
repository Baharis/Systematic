"""
peakfinders/scipy_local_max.py
==============================
An example of peak-finding algorithm that looks for morphological local maxima.

Algorithm
---------
1. Optionally uniform-smooth to suppress single hot pixels.
2. Find local maxima via maximum_filter.
3. Apply a relative and/or absolute threshold (OR logic).
4. Label connected components; take the intensity-weighted centroid of each.
5. Sort by intensity (brightest first), cap at max_peaks.
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter, label

from peakfinders.base import BasePeakFinder
from data_model import PeakResult


@dataclass
class ScipyLocalMaxVariables:
    """Tkinter variables for the ScipyLocalMax peak finder."""
    smooth_size = tk.IntVar(value=3)
    min_dist    = tk.IntVar(value=10)
    thresh_rel  = tk.DoubleVar(value=0.05)
    thresh_abs  = tk.DoubleVar(value=0.0)
    max_peaks   = tk.IntVar(value=500)


class ScipyLocalMaxFinder(BasePeakFinder):

    NAME        = 'Scipy Local Max'
    DESCRIPTION = 'Morphological local-maximum filter using scipy.ndimage'
    VARIABLES   = ScipyLocalMaxVariables

    def __init__(self) -> None:
        self.var = self.VARIABLES()

    def frame(self, parent: tk.Widget) -> tk.Frame:
        new = tk.Frame(parent)

        ttk.Label(
            new,
            text=self.DESCRIPTION,
            font=("Helvetica", 8, "italic"),
            foreground="#555",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))

        rows = [
            ("Smooth kernel size (px)",      "smooth_size", 1,    21,  2,    "{:.0f}"),
            ("Min distance (px)",            "min_dist",    1,   100,  1,    "{:.0f}"),
            ("Threshold – relative (×max)",  "thresh_rel",  0.0,  1.0, 0.005,"{:.3f}"),
            ("Threshold – absolute (ADU)",   "thresh_abs",  0.0,  1e6, 1.0,  "{:.1f}"),
            ("Max peaks",                    "max_peaks",   1,  2000,  1,    "{:.0f}"),
        ]

        for r, (label_text, attr, lo, hi, res, fmt) in enumerate(rows, start=1):
            ttk.Label(new, text=label_text, anchor="w").grid(
                row=r, column=0, sticky="w", pady=3, padx=(0, 6))

            tkvar = getattr(self.var, attr)          # the actual tk.*Var

            slider = ttk.Scale(new, variable=tkvar, from_=lo, to=hi,
                               orient=tk.HORIZONTAL, length=200)
            slider.grid(row=r, column=1, sticky="ew", padx=4)

            entry_var = tk.StringVar(value=fmt.format(tkvar.get()))
            entry = ttk.Entry(new, textvariable=entry_var, width=9)
            entry.grid(row=r, column=2, padx=4)

            # trace_add passes (var_name, index, mode) → absorb with *args
            def _s2e(*args, v=tkvar, ev=entry_var, f=fmt):
                ev.set(f.format(v.get()))

            def _e2v(*args, v=tkvar, ev=entry_var, lo=lo, hi=hi):
                try:
                    val = float(ev.get())
                    v.set(max(lo, min(hi, val)))
                except ValueError:
                    pass

            tkvar.trace_add("write", _s2e)
            entry_var.trace_add("write", _e2v)

        new.columnconfigure(1, weight=1)
        return new

    def run(self, image: np.ndarray) -> PeakResult:
        smooth_size = max(1, int(self.var.smooth_size.get()))
        min_dist    = max(1, int(self.var.min_dist.get()))
        thresh_rel  = float(self.var.thresh_rel.get())
        thresh_abs  = float(self.var.thresh_abs.get())
        max_peaks   = max(1, int(self.var.max_peaks.get()))

        params = dict(
            smooth_size=smooth_size, min_dist=min_dist,
            thresh_rel=thresh_rel, thresh_abs=thresh_abs,
            max_peaks=max_peaks,
        )

        work = uniform_filter(image, size=smooth_size) if smooth_size > 1 else image.copy()

        neighbourhood = maximum_filter(work, size=min_dist)
        local_max = work == neighbourhood

        img_max = work.max()
        mask_rel = (work > thresh_rel * img_max) if thresh_rel > 0 else np.zeros_like(local_max)
        mask_abs = (work > thresh_abs)            if thresh_abs > 0 else np.zeros_like(local_max)
        thresh_mask = mask_rel | mask_abs
        if not thresh_mask.any():
            thresh_mask = work > 0

        candidates = local_max & thresh_mask
        labelled, n_features = label(candidates)

        if n_features == 0:
            return PeakResult.empty(finder_name=self.NAME, params=params)

        rows_out, cols_out, intensities_out = [], [], []
        for i in range(1, n_features + 1):
            mask = labelled == i
            total_w = image[mask].sum()
            if total_w == 0:
                ys, xs = np.where(mask)
                cy, cx = float(ys.mean()), float(xs.mean())
            else:
                ys, xs = np.where(mask)
                cy = float((ys * image[mask]).sum() / total_w)
                cx = float((xs * image[mask]).sum() / total_w)
            ri = min(max(int(round(cy)), 0), image.shape[0] - 1)
            ci = min(max(int(round(cx)), 0), image.shape[1] - 1)
            rows_out.append(cy)
            cols_out.append(cx)
            intensities_out.append(float(image[ri, ci]))

        rows_arr = np.array(rows_out)
        cols_arr  = np.array(cols_out)
        ints_arr  = np.array(intensities_out)

        order = np.argsort(ints_arr)[::-1][:max_peaks]
        return PeakResult.from_arrays(
            rows=rows_arr[order],
            cols=cols_arr[order],
            intensities=ints_arr[order],
            finder_name=self.NAME,
            params=params,
        )
