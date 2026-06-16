"""
data_model.py
=============
Canonical data structures shared across the whole application.

PeakResult
----------
A thin wrapper around a pandas DataFrame with a fixed column schema.
All peakfinders must return a PeakResult (or a plain DataFrame with the
same columns — PeakResult.from_dataframe() can coerce it).

Required columns
~~~~~~~~~~~~~~~~
    row      float  – sub-pixel row coordinate  (image y-axis)
    col      float  – sub-pixel column coordinate (image x-axis)
    intensity float – raw (un-normalised) pixel value at the peak

Optional / peakfinder-specific columns may be appended freely;
display code ignores unknown columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical column names
# ---------------------------------------------------------------------------

REQUIRED_COLS: tuple[str, ...] = ("row", "col", "intensity")


# ---------------------------------------------------------------------------
# PeakResult
# ---------------------------------------------------------------------------

@dataclass
class PeakResult:
    """
    Immutable-ish container for the peaks found in a single image.

    Parameters
    ----------
    peaks : pd.DataFrame
        Must contain at least the columns in REQUIRED_COLS.
    source_file : Path | None
        The image file these peaks came from.
    finder_name : str
        Human-readable name of the algorithm that produced these peaks.
    params : dict
        The parameter values used when the algorithm was called.
    """

    peaks: pd.DataFrame
    source_file: Path | None = field(default=None)
    finder_name: str = field(default="unknown")
    params: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def empty(
        cls,
        source_file: Path | None = None,
        finder_name: str = "unknown",
        params: dict | None = None,
    ) -> "PeakResult":
        """Return a PeakResult with zero peaks."""
        df = pd.DataFrame(columns=list(REQUIRED_COLS))
        df = df.astype({c: float for c in REQUIRED_COLS})
        return cls(
            peaks=df,
            source_file=source_file,
            finder_name=finder_name,
            params=params or {},
        )

    @classmethod
    def from_arrays(
        cls,
        rows: Sequence[float],
        cols: Sequence[float],
        intensities: Sequence[float],
        source_file: Path | None = None,
        finder_name: str = "unknown",
        params: dict | None = None,
        extra: dict[str, Sequence] | None = None,
    ) -> "PeakResult":
        """
        Build a PeakResult from plain arrays.

        Parameters
        ----------
        rows, cols, intensities : array-like of float
        extra : optional dict of {column_name: array} for peakfinder-specific data
        """
        data: dict[str, Sequence] = {
            "row": rows,
            "col": cols,
            "intensity": intensities,
        }
        if extra:
            data.update(extra)
        df = pd.DataFrame(data)
        return cls(
            peaks=df,
            source_file=source_file,
            finder_name=finder_name,
            params=params or {},
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        source_file: Path | None = None,
        finder_name: str = "unknown",
        params: dict | None = None,
    ) -> "PeakResult":
        """Coerce an arbitrary DataFrame that has at least REQUIRED_COLS."""
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"DataFrame is missing required columns: {missing}\n"
                f"Present columns: {list(df.columns)}"
            )
        return cls(
            peaks=df.copy(),
            source_file=source_file,
            finder_name=finder_name,
            params=params or {},
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.peaks)

    @property
    def rows(self) -> np.ndarray:
        return self.peaks["row"].to_numpy()

    @property
    def cols(self) -> np.ndarray:
        return self.peaks["col"].to_numpy()

    @property
    def intensities(self) -> np.ndarray:
        return self.peaks["intensity"].to_numpy()

    def __repr__(self) -> str:
        return (
            f"PeakResult(n={len(self)}, "
            f"finder='{self.finder_name}', "
            f"file='{self.source_file and self.source_file.name}')"
        )
