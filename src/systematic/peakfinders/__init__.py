"""
peakfinders/__init__.py
=======================
Auto-discovery of BasePeakFinder subclasses.

Any .py file placed in this directory that defines a class inheriting from
BasePeakFinder is picked up automatically.  The file `base.py` itself is
excluded.  Discovery order is alphabetical by filename.

Usage
-----
    from peakfinders import discover_finders
    finders = discover_finders()   # -> list[BasePeakFinder instances]
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BasePeakFinder


def discover_finders() -> list["BasePeakFinder"]:
    """
    Return one instantiated instance of every BasePeakFinder subclass found
    in this package directory, sorted by the module filename.
    """
    from .base import BasePeakFinder  # local import avoids circularity

    found: list[BasePeakFinder] = []
    pkg_path = Path(__file__).parent

    for _finder, module_name, _is_pkg in sorted(
        pkgutil.iter_modules([str(pkg_path)])
    ):
        if module_name == "base":
            continue
        try:
            module = importlib.import_module(f"peakfinders.{module_name}")
        except Exception as exc:
            print(f"[peakfinders] Could not import '{module_name}': {exc}")
            continue

        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BasePeakFinder)
                and obj is not BasePeakFinder
                and not inspect.isabstract(obj)
            ):
                try:
                    found.append(obj())
                except Exception as exc:
                    print(f"[peakfinders] Could not instantiate {obj}: {exc}")

    return found
