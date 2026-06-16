"""
image_store.py
==============
Central, GUI-free store for loaded image data and peak results.

The store is the single source of truth:
  • raw numpy arrays are kept here (loaded once, never copied into widgets)
  • PeakResult objects for the most recent run on each image are cached here
  • GUI windows hold only a reference to the store + a filepath key

Listeners (GUI windows) can register a callback that fires whenever the
store's state changes, so they can redraw themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from data_model import PeakResult


# ---------------------------------------------------------------------------
# Entry: one loaded image
# ---------------------------------------------------------------------------

@dataclass
class ImageEntry:
    filepath: Path
    raw: np.ndarray          # 2-D float64, original ADU / count values, NOT normalised
    result: PeakResult | None = field(default=None)

    @property
    def shape(self) -> tuple[int, int]:
        return self.raw.shape   # (rows, cols)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class ImageStore:
    """Thread-safe central loaded diffraction data registry."""

    def __init__(self) -> None:
        self._entries: dict[Path, ImageEntry] = {}
        self._listeners: list[Callable[[Path, str], None]] = []

    # ------------------------------------------------------------------
    # Listener / observer pattern
    # ------------------------------------------------------------------

    def add_listener(self, cb: Callable[[Path, str], None]) -> None:
        """
        Register a callback.

        cb(path, event) where event is one of:
            "loaded"   – new image added
            "removed"  – image removed
            "result"   – peak result updated for `path`
            "cleared"  – all images removed  (path will be None)
        """
        self._listeners.append(cb)

    def remove_listener(self, cb: Callable[[Path, str], None]) -> None:
        self._listeners = [l for l in self._listeners if l is not cb]

    def _notify(self, path: Path | None, event: str) -> None:
        """Notify the listener ov en event concerning "path"-indexed image."""
        for cb in self._listeners:
            try:
                cb(path, event)
            except Exception as exc:
                print(f'[ImageStore] listener error: {exc}')

    def load(self, path: Path) -> ImageEntry:
        """Load a TIFF (or any PIL-readable format) and store its raw data."""
        if path in self._entries:
            return self._entries[path]

        with Image.open(path) as img:
            arr = np.array(img, dtype=np.int64)

        if arr.ndim != 2:
            msg = f'Expected a 2D image, got {arr.shape=} from {path.name=}'
            raise ValueError(msg)

        if arr.min() < 0:  # Ensure non-negative (some detectors encode with offsets)
            arr -= arr.min()

        entry = ImageEntry(filepath=path, raw=arr)
        self._entries[path] = entry
        self._notify(path, 'loaded')
        return entry

    def set_result(self, path: Path, result: PeakResult) -> None:
        """Store peak-finding output for a given image."""
        if path not in self._entries:
            raise KeyError(f'{path} is not loaded in the store.')
        self._entries[path].result = result
        self._notify(path, 'result')

    def remove(self, path: Path) -> None:
        if path in self._entries:
            del self._entries[path]
            self._notify(path, 'removed')

    def clear(self) -> None:
        self._entries.clear()
        self._notify(None, 'cleared')

    def __contains__(self, path: Path) -> bool:
        return path in self._entries

    def __getitem__(self, path: Path) -> ImageEntry:
        return self._entries[path]

    def __len__(self) -> int:
        return len(self._entries)

    def paths(self) -> list[Path]:
        return list(self._entries.keys())

    def entries(self) -> list[ImageEntry]:
        return list(self._entries.values())

    def get_result(self, path: Path) -> PeakResult | None:
        entry = self._entries.get(path)
        return entry.result if entry else None
