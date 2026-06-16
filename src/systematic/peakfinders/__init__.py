"""
peakfinders/__init__.py
=======================
Auto-discovery of peakfinder plugins.

Any .py file in this directory whose module contains a class with all three
of (NAME, frame, run) is picked up automatically.  Inheriting BasePeakFinder
is not required.  Discovery order is alphabetical by filename.

Usage
-----
    from peakfinders import discover_finders
    finders = discover_finders()   # -> list of finder instances
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path

from .base import BasePeakFinder


def _is_peakfinder_class(cls) -> bool:
    return issubclass(cls, BasePeakFinder) and not inspect.isabstract(cls)


def discover_peakfinders() -> list:
    """Return 1 instance of every peakfinder class found in this package dir."""
    found = []
    pkg_path = str(Path(__file__).parent)

    for _finder, module_name, _is_pkg in sorted(pkgutil.iter_modules([pkg_path])):
        try:
            module = importlib.import_module(f'peakfinders.{module_name}')
        except Exception as exc:
            print(f"[peakfinders] Could not import '{module_name}': {exc}")
            continue

        for _name, obj in inspect.getmembers(module, inspect.isclass):

            if obj.__module__ != module.__name__:
                continue
            if _is_peakfinder_class(obj):
                try:
                    found.append(obj())
                except Exception as exc:
                    print(f"[peakfinders] Could not instantiate {obj}: {exc}")

    return found
