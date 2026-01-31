from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .filtration import FiltrationOrder
from .persistence import PersistencePairs


@dataclass(frozen=True)
class Barcode:
    r"""
    Persistence barcode, grouped by homological degree.

    For each integer degree ``d``, a barcode is a multiset of intervals

    .. math::

        [b,\, d) \subseteq \mathbb{R} \cup \{\infty\},

    where ``b`` is the birth time and ``d`` is the death time of a homology class
    in degree ``d`` along a filtration.

    This class stores, for every degree, the corresponding list of intervals
    written as pairs ``(birth, death)`` with ``death = None`` representing
    an infinite endpoint.

    Attributes
    ----------
    intervals_by_dim:
        Dictionary ``dim ↦ [(birth, death), …]``.
        Each ``birth`` is a finite float.
        Each ``death`` is either a finite float or ``None``.
    """

    intervals_by_dim: Dict[int, List[Tuple[float, Optional[float]]]]


def barcodes_from_persistence(filtration: FiltrationOrder, persistence: PersistencePairs) -> Barcode:
    r"""
    Convert persistence pairing data into a barcode.

    Let ``filtration`` be an ordering of simplices with associated filtration
    values ``v_i`` indexed by the filtration position ``i``. Let ``persistence``
    provide

    - paired indices ``(birth_i, death_i, dim)`` encoding a class in degree ``dim``
      born at index ``birth_i`` and killed at index ``death_i``
    - unpaired indices ``(birth_i, dim)`` encoding an essential class in degree
      ``dim`` with infinite death time

    This function returns the barcode that records the corresponding time
    intervals

    .. math::

        \bigl(v_{\mathrm{birth}},\, v_{\mathrm{death}}\bigr)
        \quad \text{or} \quad
        \bigl(v_{\mathrm{birth}},\, \infty\bigr).

    Sorting convention
    ------------------
    Within each degree, intervals are sorted lexicographically by
    ``(birth, death)``, treating ``None`` as ``+∞``. This is a presentation choice
    and does not change the underlying multiset.

    Parameters
    ----------
    filtration:
        Filtration order with array ``values`` such that ``values[i]`` is the
        filtration value at index ``i``.
    persistence:
        Persistence output containing paired and unpaired indices.

    Returns
    -------
    Barcode
        Barcode grouped by degree.

    Notes
    -----
    This conversion is purely representational. It assumes that indices in
    ``persistence`` refer to valid positions in ``filtration.values``.
    """
    values = filtration.values
    intervals_by_dim: Dict[int, List[Tuple[float, Optional[float]]]] = {}

    # Finite intervals from paired births and deaths.
    for birth_i, death_i, dim in persistence.pairs:
        intervals_by_dim.setdefault(dim, []).append((float(values[birth_i]), float(values[death_i])))

    # Infinite intervals from unpaired births.
    for birth_i, dim in persistence.unpaired:
        intervals_by_dim.setdefault(dim, []).append((float(values[birth_i]), None))

    # Canonical ordering for display: (birth ascending, death ascending, ∞ last).
    for dim in intervals_by_dim:
        intervals_by_dim[dim].sort(key=lambda x: (x[0], float("inf") if x[1] is None else x[1]))

    return Barcode(intervals_by_dim=intervals_by_dim)