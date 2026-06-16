"""
peakfinders/base.py
===================
Protocol / abstract base class that every peakfinder plugin must satisfy.

How to add a new peakfinder
---------------------------
1.  Create a new .py file inside the `peakfinders/` directory.
2.  Define a class that has the following interface (inheriting BasePeakFinder
    is optional but recommended for IDE support and isinstance checks):

        NAME        : str   – short display name used for the tab label
        DESCRIPTION : str   – one-line description shown as a subtitle
        VARIABLES   : type  – a dataclass whose fields are tk.*Var instances;
                              one instance is created as self.var in __init__
        frame(parent) -> tk.Frame
                      – build and return a tk.Frame containing all parameter
                        widgets, wired to self.var
        run(image)  -> PeakResult
                      – read self.var, run the algorithm, return a PeakResult

Convention for VARIABLES dataclass
------------------------------------
    @dataclass
    class MyVars:
        threshold = tk.DoubleVar(value=0.1)
        min_dist  = tk.IntVar(value=10)

    class MyFinder(BasePeakFinder):
        VARIABLES = MyVars
        def __init__(self):
            self.var = self.VARIABLES()

Access values in run() as:  self.var.threshold.get()

Discovery
---------
peakfinders/__init__.py scans this package automatically.  Any class that
has NAME, frame, and run attributes is picked up — no explicit registration
needed.  Inheriting BasePeakFinder is not required for discovery.
"""

from __future__ import annotations

import tkinter as tk
from abc import ABC, abstractmethod

import numpy as np

from data_model import PeakResult


class BasePeakFinder(ABC):
    """
    Optional base class for peakfinder plugins.

    Inherit from this for IDE autocompletion and abstract-method checking.
    Discovery does not require inheritance — duck-typing is used instead.
    """

    NAME: str = "Unnamed"
    DESCRIPTION: str = ""

    @abstractmethod
    def frame(self, parent: tk.Widget) -> tk.Frame:
        """
        Build parameter widgets and return a tk.Frame.

        The frame will be embedded directly into the notebook tab.
        Wire all widgets to variables stored on self (e.g. self.var.*).
        """

    @abstractmethod
    def run(self, image: np.ndarray) -> PeakResult:
        """
        Run peak-finding on a single image.

        Parameters
        ----------
        image : 2-D float64 ndarray, original (un-normalised) pixel values,
                guaranteed non-negative.

        Returns
        -------
        PeakResult  – filepath and finder_name are filled in by the caller.
        """
