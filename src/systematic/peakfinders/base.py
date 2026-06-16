"""
peakfinders/base.py
===================
Protocol / abstract base class that every peakfinder plugin must satisfy.

How to add a new peakfinder
---------------------------
1.  Create a new .py file inside the `peakfinders/` directory.
2.  Define a class that inherits from ``BasePeakFinder``.
3.  Implement the three required members:
        NAME        – short display name used for the tab label
        DESCRIPTION – one-line description shown as a tooltip / subtitle
        build_ui()  – populate a ttk.LabelFrame with your parameter widgets
        run()       – read your widgets, run the algorithm, return a PeakResult

The main window discovers peakfinders automatically by scanning this package;
no registration step is needed.

Parameter widget conventions
-----------------------------
Use standard tkinter variables (tk.DoubleVar, tk.IntVar, tk.StringVar).
Store them as instance attributes so ``run()`` can read them.
You may use any tkinter/ttk widgets you like inside ``build_ui()``.
"""

from __future__ import annotations

import tkinter as tk
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from data_model import PeakResult


class BasePeakFinder(ABC):
    """
    Abstract base class for peakfinder plugins.

    Subclasses must define:
        NAME        : str            – Short name shown on the notebook tab
        DESCRIPTION : str            – One-line description shown below the tab
        build_ui()                   – create parameter widgets in `parent`
        run()                        – execute algorithm, return PeakResult
    """

    NAME: str = "Unnamed"
    DESCRIPTION: str = ""

    def __init__(self) -> None:
        # Subclasses should call super().__init__() and then create
        # tk.*Var instances as instance attributes.
        pass

    @abstractmethod
    def build_ui(self, parent: tk.Widget) -> None:
        """
        Populate `parent` (a ttk.LabelFrame) with parameter widgets.

        `parent` is freshly created for each peakfinder tab;
        the peakfinder owns it entirely.
        """

    @abstractmethod
    def run(
        self,
        image: np.ndarray,
        filepath: Path,
    ) -> PeakResult:
        """
        Run peak-finding on a single image.

        Parameters
        ----------
        image    : 2-D float64 array, original (un-normalised) pixel values.
                   Non-negative; may contain zeros.
        filepath : Path to the source file (for PeakResult metadata).

        Returns
        -------
        PeakResult
        """
