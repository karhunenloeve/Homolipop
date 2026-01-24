from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .alpha import AlphaFiltration
from .simplices import Simplex, simplex_dim


@dataclass(frozen=True)
class FiltrationOrder:
    simplices: List[Simplex]
    values: List[float]


def alpha_filtration_order(alpha: AlphaFiltration, simplices: Sequence[Simplex]) -> FiltrationOrder:
    alpha_sq: Dict[Simplex, float] = alpha.alpha_sq

    ordered_simplices = sorted(simplices, key=lambda s: (alpha_sq[s], simplex_dim(s), s))
    ordered_values = [alpha_sq[s] for s in ordered_simplices]

    return FiltrationOrder(simplices=ordered_simplices, values=ordered_values)