"""
image_window.py
===============
Display-only Toplevel window for a single diffraction image.

Responsibilities
----------------
• Render the raw image using log1p (no prior normalisation) so the full
  dynamic range of diffraction data is visible.
• Overlay peak markers from the most recent PeakResult stored in the
  ImageStore for this file.
• When an EvalResult is available, draw nearest-neighbour lines between
  peaks coloured green→yellow→red according to how well each inter-peak
  distance matches the supplied powder pattern.
• Print the match score in the bottom-right corner of the image.
• Listen to the ImageStore and redraw automatically when results update.

The window holds NO raw data itself — it only reads from ImageStore.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.lines import Line2D
import tkinter as tk

from image_store import ImageStore
from data_model import PeakResult
from peak_eval import PairwiseEval


class ImageWindow:
    """
    One Toplevel per loaded TIFF.

    Parameters
    ----------
    parent   : tk.Misc          – parent widget (the main Tk root)
    store    : ImageStore        – shared data store
    filepath : Path              – key into the store
    on_close : callable(Path)    – called when the user closes the window
    """

    def __init__(
        self,
        parent: tk.Misc,
        store: ImageStore,
        filepath: Path,
        on_close: Callable[[Path], None] | None = None,
    ) -> None:
        self._store        = store
        self._filepath     = filepath
        self._on_close_cb  = on_close
        self._eval: PairwiseEval | None = None   # set externally via update_eval()

        self._build_window(parent)
        self._draw()

        self._store.add_listener(self._on_store_event)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_window(self, parent: tk.Misc) -> None:
        self.win = tk.Toplevel(parent)
        self.win.title(f'Image – {self._filepath.name}')
        self.win.geometry('720x620')
        self.win.protocol('WM_DELETE_WINDOW', self._on_close)

        # Status bar (bottom)
        self._status_var = tk.StringVar(value='')
        tk.Label(
            self.win,
            textvariable=self._status_var,
            anchor='w',
            relief=tk.SUNKEN,
            bd=1,
            font=('Courier', 9),
        ).pack(side=tk.BOTTOM, fill=tk.X)

        # Matplotlib figure
        self._fig = Figure(figsize=(6.5, 5.5), dpi=100)
        self._ax  = self._fig.add_subplot(111)
        self._fig.tight_layout(pad=1.5)

        self._canvas = FigureCanvasTkAgg(self._fig, master=self.win)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        tb_frame = tk.Frame(self.win)
        tb_frame.pack(fill=tk.X)
        self._toolbar = NavigationToolbar2Tk(self._canvas, tb_frame)
        self._toolbar.update()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self) -> None:
        ax = self._ax
        ax.clear()

        entry = self._store[self._filepath]
        raw   = entry.raw                  # 2-D float64, original counts

        # log1p for display — no normalisation; preserves dynamic range
        ax.imshow(
            np.log1p(raw),
            cmap='inferno',
            origin='upper',
            aspect='equal',
            interpolation='nearest',
        )

        result: PeakResult | None = entry.result
        n_peaks = 0

        if result is not None and len(result) > 0:
            n_peaks = len(result)
            rows = result.rows
            cols = result.cols

            # ── Peak markers ─────────────────────────────────────────
            ax.scatter(
                cols, rows,
                s=50,
                facecolors='none',
                edgecolors='cyan',
                linewidths=1.2,
                zorder=5,
                label=f'{n_peaks} peaks  [{result.finder_name}]',
            )

            # ── Eval overlay ──────────────────────────────────────────
            ev = self._eval
            if ev is not None and len(ev.pairs) > 0:
                self._draw_eval_lines(ax, rows, cols, ev)
                self._draw_score_annotation(ax, raw.shape, ev.score)

            ax.legend(
                loc='upper right',
                fontsize=8,
                framealpha=0.55,
                labelcolor='white',
                facecolor='#1a1a1a',
                edgecolor='#444',
            )

        # ── Aesthetics ────────────────────────────────────────────────
        self._fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='#aaa', labelsize=7)
        ax.set_title(self._filepath.name, fontsize=9, color='#ccc', pad=4)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

        h, w   = raw.shape
        finder = result.finder_name if result else '—'
        score_str = f'  |  score: {self._eval.score:.3f}' if self._eval else ''
        self._status_var.set(
            f'{w}×{h} px  |  peaks: {n_peaks}  |  finder: {finder}{score_str}'
        )

        self._canvas.draw_idle()

    def _draw_eval_lines(
        self,
        ax,
        rows: np.ndarray,
        cols: np.ndarray,
        ev: PairwiseEval,
    ) -> None:
        """
        Draw nearest-neighbour lines between peaks, coloured by match quality.

        Each line segment connects peak pair (i, j).  The colour runs from
        green (rel_error ≈ 0, good match) through yellow to red (bad match).
        Lines are drawn below the peak markers (zorder=3).
        """
        for k, (i, j) in enumerate(ev.pairs):
            rgba  = ev.colors[k]              # (4,) float32
            x     = [cols[i], cols[j]]
            y     = [rows[i], rows[j]]
            ax.add_line(Line2D(
                x, y,
                color=tuple(float(c) for c in rgba),
                linewidth=1.2,
                alpha=0.85,
                zorder=3,
                solid_capstyle='round',
            ))

    def _draw_score_annotation(
        self,
        ax,
        image_shape: tuple[int, int],
        score: float,
    ) -> None:
        """
        Print the match score in the bottom-right corner of the image axes.

        The text colour transitions from green (score=1) through yellow to
        red (score=0) to match the line colours.
        """
        # Interpolate colour: red at 0, yellow at 0.5, green at 1
        s = float(np.clip(score, 0.0, 1.0))
        if s >= 0.5:
            t = (s - 0.5) * 2.0
            text_color = (1.0 - t, 1.0, 0.0)       # yellow → green
        else:
            t = s * 2.0
            text_color = (1.0, t, 0.0)              # red → yellow

        h, w = image_shape
        ax.text(
            w - 6, h - 6,
            f'score: {score:.3f}',
            ha='right', va='bottom',
            fontsize=10,
            fontweight='bold',
            color=text_color,
            zorder=10,
            bbox=dict(
                boxstyle='round,pad=0.25',
                facecolor='#1a1a1a',
                edgecolor='none',
                alpha=0.65,
            ),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def lift_to_front(self) -> None:
        self.win.lift()
        self.win.focus_set()

    def update_eval(self, ev: PairwiseEval | None) -> None:
        """
        Receive a new evaluation result from the main window and redraw.

        Called by MainWindow._eval_peaks() after computing PairwiseEval
        for this image.  Passing None clears any previous overlay.
        """
        self._eval = ev
        self._draw()

    def clear_eval(self) -> None:
        """Remove any eval overlay and redraw."""
        self.update_eval(None)

    # ------------------------------------------------------------------
    # Store listener
    # ------------------------------------------------------------------

    def _on_store_event(self, path: Path | None, event: str) -> None:
        if event == 'cleared' or (path == self._filepath and event == 'removed'):
            self._on_close()
            return
        if path == self._filepath and event == 'result':
            # New peak result: clear stale eval overlay, redraw
            self._eval = None
            self._draw()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        try:
            self._store.remove_listener(self._on_store_event)
        except Exception:
            pass
        if self._on_close_cb:
            self._on_close_cb(self._filepath)
        try:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self.win.destroy()
        except Exception:
            pass
