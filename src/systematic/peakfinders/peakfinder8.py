from __future__ import annotations

import tkinter as tk
from dataclasses import asdict, dataclass
from tkinter import ttk

import numpy as np
from peakfinder8 import peakfinder8

from peakfinders.base import BasePeakFinder
from data_model import PeakResult


@dataclass
class PeakFinder8Variables:
    """Tkinter variables for the ScipyLocalMax peak finder."""
    adc_thresh      = tk.IntVar(value=40)
    min_snr         = tk.IntVar(value=4)
    min_pix_count   = tk.IntVar(value=1)
    max_pix_count   = tk.IntVar(value=40)
    local_bg_radius = tk.IntVar(value=3)


class PeakFinder8Finder(BasePeakFinder):

    NAME = "PeakFinder8"
    DESCRIPTION = 'peakfinder8 algorithm from Cheetah, CrystFEL, OnDA, diffractem,... '
    VARIABLES   = PeakFinder8Variables

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
            ('Count threshold',         'adc_thresh',       1,  100, 1, '{:.0f}'),
            ('Min sigma-to-noise',      'min_snr',          1,  100, 1, '{:.0f}'),
            ('Minimum pixel count',     'min_pix_count',    1,  1e4, 1, '{:.0f}'),
            ('Maximum pixel count',     'max_pix_count',    1,  1e4, 1, '{:.0f}'),
            ('Local background radius', 'local_bg_radius',  1,  100, 1, '{:.0f}'),
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
        """Run peakfinder8 algorithm."""
        y0 = image.shape[0] / 2.0
        x0 = image.shape[1] / 2.0
        yy, xx = np.indices(image.shape, dtype=np.float32)
        rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2).astype(np.float32)
        mask = np.ones_like(image, dtype=np.int8)
        mask[rr > 1000] = 0
        mask[rr < 10] = 0

        peaks = peakfinder8(
            500,  # max peaks
            image.astype(np.float32, copy=False),
            mask,
            rr,
            image.shape[1],
            image.shape[0],
            1,
            1,
            self.var.adc_thresh.get(),
            self.var.min_snr.get(),
            self.var.min_pix_count.get(),
            self.var.max_pix_count.get(),
            self.var.local_bg_radius.get(),
        )
        return PeakResult.from_arrays(
            rows=peaks[1],
            cols=peaks[0],
            intensities=peaks[2],
            finder_name=self.NAME,
            params=asdict(self.var),
        )