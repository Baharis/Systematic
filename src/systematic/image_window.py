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
• Listen to the ImageStore and redraw automatically when a new result
  arrives or the image is removed.

The window holds NO raw data itself — it only reads from ImageStore.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import LogNorm

from image_store import ImageStore
from data_model import PeakResult


class ImageWindow:
    """
    One Toplevel per loaded TIFF.

    Parameters
    ----------
    parent   : tk.Tk or tk.Toplevel  – parent widget
    store    : ImageStore             – shared data store
    filepath : Path                   – key into the store
    on_close : callable(Path)         – called when user closes the window
    """

    def __init__(
        self,
        parent: tk.Misc,
        store: ImageStore,
        filepath: Path,
        on_close: "Callable[[Path], None]" = None,
    ) -> None:
        self._store = store
        self._filepath = filepath
        self._on_close_cb = on_close

        self._build_window(parent)
        self._draw()

        # Subscribe to store events
        self._store.add_listener(self._on_store_event)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_window(self, parent: tk.Misc) -> None:
        self.win = tk.Toplevel(parent)
        self.win.title(f"Image – {self._filepath.name}")
        self.win.geometry("720x620")
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Status bar ───────────────────────────────────────────────
        self._status_var = tk.StringVar(value="")
        status = tk.Label(
            self.win,
            textvariable=self._status_var,
            anchor="w",
            relief=tk.SUNKEN,
            bd=1,
            font=("Courier", 9),
        )
        status.pack(side=tk.BOTTOM, fill=tk.X)

        # ── Matplotlib figure ────────────────────────────────────────
        self._fig = Figure(figsize=(6.5, 5.5), dpi=100)
        self._ax = self._fig.add_subplot(111)
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
        self._ax.clear()

        entry = self._store[self._filepath]
        raw = entry.raw                          # 2-D float64, original counts

        # log1p transform for display – no normalisation, preserving relative
        # intensities across the full dynamic range of diffraction data
        display = np.log1p(raw)

        self._ax.imshow(
            display,
            cmap="inferno",
            origin="upper",
            aspect="equal",
            interpolation="nearest",
        )

        # ── Peak overlay ─────────────────────────────────────────────
        result: PeakResult | None = entry.result
        n_peaks = 0

        if result is not None and len(result) > 0:
            n_peaks = len(result)
            self._ax.scatter(
                result.cols,         # col → x-axis
                result.rows,         # row → y-axis
                s=50,
                facecolors="none",
                edgecolors="cyan",
                linewidths=1.2,
                zorder=5,
                label=f"{n_peaks} peaks  [{result.finder_name}]",
            )
            legend = self._ax.legend(
                loc="upper right",
                fontsize=8,
                framealpha=0.55,
                labelcolor="white",
                facecolor="#1a1a1a",
                edgecolor="#444",
            )

        # ── Aesthetics ───────────────────────────────────────────────
        self._fig.patch.set_facecolor("#1a1a1a")
        self._ax.set_facecolor("#1a1a1a")
        self._ax.tick_params(colors="#aaa", labelsize=7)
        self._ax.set_title(
            self._filepath.name, fontsize=9, color="#ccc", pad=4
        )
        for spine in self._ax.spines.values():
            spine.set_edgecolor("#444")

        h, w = raw.shape
        finder_name = result.finder_name if result else "—"
        self._status_var.set(
            f"{w}×{h} px  |  peaks: {n_peaks}  |  finder: {finder_name}"
        )

        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Store listener
    # ------------------------------------------------------------------

    def _on_store_event(self, path: Path | None, event: str) -> None:
        """React to store changes that affect this window."""
        if event == "cleared" or (path == self._filepath and event == "removed"):
            self._on_close()
            return
        if path == self._filepath and event == "result":
            self._draw()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def lift_to_front(self) -> None:
        self.win.lift()
        self.win.focus_set()

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
