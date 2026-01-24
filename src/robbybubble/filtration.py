from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .alpha import AlphaFiltration
from .simplices import Simplex, simplex_dim


@dataclass(frozen=True)
class FiltrationOrder:
    """
    A total order of simplices compatible with a filtration.

    Invariants
    - simplices[i] appears no later than simplices[j] whenever its filtration key is smaller.
    - values[i] equals the filtration value of simplices[i].
    """
    simplices: List[Simplex]
    values: List[float]


def alpha_filtration_order(alpha: AlphaFiltration, simplices: Sequence[Simplex]) -> FiltrationOrder:
    """
    Sort simplices into a filtration order compatible with alpha^2 values.

    Ordering key
    1. alpha_sq(s) ascending
    2. dimension(s) ascending
    3. lexicographic vertex order ascending

    Correctness
    - The dimension tie-break ensures that if face and coface share the same alpha value,
      the face is ordered first since dim(face) < dim(coface).
    - Lex order makes the sort total and deterministic.

    Optimality
    - Any comparison-based explicit ordering of N simplices requires Omega(N log N) comparisons.
      This routine attains that bound up to constant factors.
    """
    alpha_sq: Dict[Simplex, float] = alpha.alpha_sq

    ordered_simplices = sorted(
        simplices,
        key=lambda s: (alpha_sq[s], simplex_dim(s), s),
    )

    ordered_values = [alpha_sq[s] for s in ordered_simplices]
    return FiltrationOrder(simplices=ordered_simplices, values=ordered_values)