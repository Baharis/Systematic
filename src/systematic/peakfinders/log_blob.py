"""
peakfinders/log_blob.py
=======================
Peak finder #2 – Laplacian of Gaussian (LoG) blob detector.

Algorithm
---------
1. Compute the scale-normalised LoG response:
       R = σ² · ∇²(Gσ * I)
   approximated as the difference of two Gaussian smoothings (DoG):
       DoG(σ, k) = G(σ·k) * I  −  G(σ) * I
2. Threshold the *negative* LoG response (bright blobs produce minima in LoG).
3. Find local minima of the response map within a (min_distance) window.
4. Return up to max_peaks peaks sorted by response strength.

Why DoG instead of true LoG?
  scipy.ndimage.gaussian_laplace is available but DoG is faster and the
  two are equivalent up to a scale factor.
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter, label

from peakfinders.base import BasePeakFinder
from data_model import PeakResult


@dataclass
class LogBlobVariables:
    """A dataclass with tkinter variables of the LogBlob peak finder."""

    sigma = tk.DoubleVar(value=2.0)
    dog_ratio = tk.DoubleVar(value=1.6)  # k in σ·k
    thresh_rel = tk.DoubleVar(value=0.05)  # fraction of max |response|
    min_dist = tk.IntVar(value=8)
    max_peaks = tk.IntVar(value=500)


class LogBlobPeakFinder:
    """Instance of tkFrame built into the GUI that runs LoG blob peak finder."""

    NAME = "LoG Blob"
    DESCRIPTION = "Laplacian-of-Gaussian blob detector (DoG approximation)"
    VARIABLES = LogBlobVariables

    def __init__(self) -> None:
        self.var = self.VARIABLES()

    def frame(self, parent, *args, **kwargs) -> tk.Frame:
        new = tk.Frame(parent, *args, **kwargs)

        ttk.Label(
            new,
            text=self.DESCRIPTION,
            font=("Helvetica", 8, "italic"),
            foreground="#555",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))

        rows = [
            ("σ – inner Gaussian (px)",        'sigma',      0.5,  20.0,   0.5, '{:.1f}'),
            ("k – DoG scale ratio (>1)",       'dog_ratio',  1.05,  4.0,  0.05, '{:.2f}'),
            ("Response threshold (× max |R|)", 'thresh_rel', 0.0,   1.0, 0.005, '{:.3f}'),
            ("Min peak distance (px)",         'min_dist',   1,     100,     1, '{:.0f}'),
            ("Max peaks",                      'max_peaks',  1,    2000,     1, '{:.0f}'),
        ]

        for r, (label_text, var, lo, hi, res, fmt) in enumerate(rows, start=1):
            lb = ttk.Label(new, text=label_text, anchor='w')
            lb.grid(row=r, column=0, sticky='w', pady=3, padx=(0, 6))

            s = ttk.Scale(new, variable=var, from_=lo, to=hi, orient=tk.HORIZONTAL, length=200)
            s.grid(row=r, column=1, sticky='ew', padx=4)

            var = getattr(self.var, var).get()
            entry_var = tk.StringVar(value=fmt.format(var))
            entry = ttk.Entry(new, textvariable=entry_var, width=9)
            entry.grid(row=r, column=2, padx=4)

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

        new.columnconfigure(1, weight=1)
        return new

    def run(self, image: np.ndarray) -> PeakResult:
        """Runs LoG blob peak finder on an image given variables from var object."""

        sigma = max(0.5, float(self.var.sigma.get()))
        k = max(1.05, float(self.var.dog_ratio.get()))
        thresh_rel = float(self.var.thresh_rel.get())
        min_dist = max(1, int(self.var.min_dist.get()))
        max_peaks = max(1, int(self.var.max_peaks.get()))

        params = dict(sigma=sigma, dog_ratio=k,
                      thresh_rel=thresh_rel, min_dist=min_dist,
                      max_peaks=max_peaks)

        # 1. DoG response (approximates −LoG up to a constant)
        g1 = gaussian_filter(image, sigma=sigma)
        g2 = gaussian_filter(image, sigma=sigma * k)
        dog = g2 - g1          # negative for bright blobs

        # 2. We want *negative* extrema → find minima of `dog`
        neg_dog = -dog         # now bright blobs are positive

        # 3. Threshold
        resp_max = neg_dog.max()
        if resp_max == 0:
            return PeakResult.empty(source_file=filepath,
                                    finder_name=self.NAME, params=params)
        candidates_mask = neg_dog > thresh_rel * resp_max

        # 4. Local maxima of neg_dog within min_dist window
        local_max_val = minimum_filter(-neg_dog, size=min_dist)
        local_max_mask = (-neg_dog == local_max_val) & candidates_mask

        labelled, n_features = label(local_max_mask)
        if n_features == 0:
            return PeakResult.empty(source_file=filepath,
                                    finder_name=self.NAME, params=params)

        rows_out, cols_out, intensities_out, responses_out = [], [], [], []
        for i in range(1, n_features + 1):
            ys, xs = np.where(labelled == i)
            # Centroid
            cy = float(ys.mean())
            cx = float(xs.mean())
            ri, ci = int(round(cy)), int(round(cx))
            ri = min(max(ri, 0), image.shape[0] - 1)
            ci = min(max(ci, 0), image.shape[1] - 1)
            rows_out.append(cy)
            cols_out.append(cx)
            intensities_out.append(float(image[ri, ci]))
            responses_out.append(float(neg_dog[ri, ci]))

        rows_arr = np.array(rows_out)
        cols_arr = np.array(cols_out)
        ints_arr = np.array(intensities_out)
        resp_arr = np.array(responses_out)

        # Sort by LoG response strength
        order = np.argsort(resp_arr)[::-1][:max_peaks]

        return PeakResult.from_arrays(
            rows=rows_arr[order],
            cols=cols_arr[order],
            intensities=ints_arr[order],
            source_file=filepath,
            finder_name=self.NAME,
            params=params,
            extra={"log_response": resp_arr[order]},
        )
