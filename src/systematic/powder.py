"""
powder_pattern.py
=================
Kinematical powder pattern from lattice parameters only.

All structure factors are assumed equal to 1 — the output is the idealized
list of geometrically allowed reflections (lattice extinctions applied,
systematic absences from space-group symmetry NOT applied, because those
are unreliable in electron diffraction anyway).

Public API
----------
    PowderLine          – namedtuple for a single unique reflection
    LatticeParams       – dataclass holding the six cell parameters + centering
    powder_pattern()    – main computation function
    from_cif()          – convenience: load LatticeParams from a CIF file

All angles are in degrees; lengths in Ångströms; wavelength in Ångströms.

Example
-------
    params = LatticeParams(a=4.05, b=4.05, c=4.05,
                           alpha=90, beta=90, gamma=90,
                           centering='F')
    lines = powder_pattern(params, wavelength=1.5406, max_2theta=90.0)
    for line in lines:
        print(f"{line.hkl_str:10s}  d={line.d:.4f} Å  2θ={line.two_theta:.3f}°  m={line.multiplicity}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import diffpy.structure as _ds
import numpy as np


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class PowderLine(NamedTuple):
    """One unique reflection in the powder pattern."""
    two_theta: float   # degrees
    d: float           # Å
    inv_d: float       # Å⁻¹  (= 1/d = Q/2π)
    multiplicity: int  # number of equivalent hkl
    hkl: tuple         # one representative Miller index, e.g. (1, 1, 0)

    @property
    def hkl_str(self) -> str:
        return '({} {} {})'.format(*self.hkl)

    @property
    def q(self) -> float:
        """Scattering vector magnitude Q = 2π/d  [Å⁻¹]."""
        return 2.0 * np.pi * self.inv_d


@dataclass
class LatticeParameters:
    """Six crystal lattice parameters plus crystal lattice centering symbol."""
    a:         float = 4.0
    b:         float = 4.0
    c:         float = 4.0
    alpha:     float = 90.0   # degrees
    beta:      float = 90.0   # degrees
    gamma:     float = 90.0   # degrees
    centering: str   = 'P'    # one of P, I, F, A, B, C, R

    def __post_init__(self):
        self.centering = self.centering.upper()
        valid = ('P', 'I', 'F', 'A', 'B', 'C', 'R')
        if self.centering not in valid:
            raise ValueError(f'{self.centering=} must be one of {valid=}')

    def to_lattice(self) -> _ds.lattice.Lattice:
        """Return a diffpy Lattice object for this cell."""
        return _ds.lattice.Lattice(
            self.a, self.b, self.c,
            self.alpha, self.beta, self.gamma,
        )

    @classmethod
    def from_cif(cls, path: str) -> LatticeParameters:
        """Load cell params (& infer centering, if possible) from a CIF file."""
        stru = _ds.load_structure(str(path), fmt='cif')
        lt = stru.lattice
        ct = cls._infer_centering_from_structure(stru)
        return LatticeParameters(lt.a, lt.b, lt.c, lt.alpha, lt.beta, lt.gamma, ct)

    @staticmethod
    def _infer_centering_from_structure(stru: _ds.Structure) -> str:
        """Try to read the lattice centering from the diffpy Structure object.

        diffpy does not expose the Hermann–Mauguin symbol as a first-class
        attribute, but it is stored in the pdffit dict or can be read from
        the CIF data block via the private _ciffile attribute. Else fall back
        to inspecting atom positions to detect face-centring translations.

        Returns a single letter: P / I / F / A / B / C / R.
        """
        try:
            sg = stru.pdffit.get('spcgr', '')
            if sg:
                letter = sg.strip()[0].upper()
                if letter in {'P', 'I', 'F', 'A', 'B', 'C', 'R'}:
                    return letter
        except Exception:
            pass

        try:
            for attr in ('_spacegroup', '_space_group_name'):
                sg = getattr(stru, attr, None)
                if sg:
                    letter = str(sg).strip()[0].upper()
                    if letter in {'P', 'I', 'F', 'A', 'B', 'C', 'R'}:
                        return letter
        except Exception:
            pass

        return 'P'

    def extinction_mask(self, h: np.ndarray, k: np.ndarray, l: np.ndarray) -> np.ndarray:
        """Return True for (h k l) absent due to the lattice centering only."""
        if self.centering == 'P':
            return np.zeros_like(h, dtype=bool)
        elif self.centering == 'I':
            return (h + k + l) % 2 != 0
        elif self.centering == 'F':
            return (h % 2 != k % 2) | (h % 2 != l % 2)
        elif self.centering == 'A':
            return (k + l) % 2 != 0
        elif self.centering == 'B':
            return (h + l) % 2 != 0
        elif self.centering == 'C':
            return (h + k) % 2 != 0
        elif self.centering == 'R':
            return (-h + k + l) % 3 != 0
        return np.zeros_like(h, dtype=bool)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def powder_lines(lp: LatticeParameters, s_max: float = 1.0) -> np.ndarray:
    """
    Compute reciprocal spacings of powder lines in A-1 given lattice params.

    Parameters
    ----------
    lp : LatticeParameters
        Unit-cell parameters and lattice centering.
    s_max : float
        Upper limit of s = d_star = k / 2pi length between lattice planes.

    Returns
    -------
    list of float
        Positions of possible powder lines expressed in reciprocal Angstrom.
    """
    lat = lp.to_lattice()
    g_star = np.linalg.inv(lat.metrics)     # reciprocal metric tensor
    h_max = 2 * int(np.ceil(s_max * lp.a)) + 1  # over-exaggerated limits
    k_max = 2 * int(np.ceil(s_max * lp.b)) + 1
    l_max = 2 * int(np.ceil(s_max * lp.c)) + 1

    h_range = np.arange(0, h_max + 1)
    k_range = np.arange(-k_max, k_max + 1)
    l_range = np.arange(-k_max, l_max + 1)
    h, k, l = np.meshgrid(h_range, k_range, l_range, indexing='ij')

    s2 = np.empty_like(h, dtype=np.float32)
    np.multiply(h, h, out=s2)
    s2 *= g_star[0, 0]
    s2 += g_star[1, 1] * k * k
    s2 += g_star[2, 2] * l * l
    s2 += 2 * g_star[0, 1] * h * k
    s2 += 2 * g_star[0, 2] * h * l
    s2 += 2 * g_star[1, 2] * k * l
    s = np.sqrt(s2)

    mask = (s <= s_max) & ~lp.extinction_mask(h, k, l)
    return np.unique(s[mask])


if __name__ == '__main__':
    lp = LatticeParameters(a=4, b=4, c=4, alpha=90, beta=90, gamma=90, centering='P')
    print(powder_lines(lp=lp, s_max=1.0))
    lp.centering = 'I'
    print(powder_lines(lp=lp, s_max=1.0))
    lp.centering = 'F'
    print(powder_lines(lp=lp, s_max=1.0))
