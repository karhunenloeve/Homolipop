from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, TypeVar

import numpy as np

from .alpha import AlphaFiltration, alpha_values_squared
from .barcodes import Barcode, barcodes_from_persistence
from .filtration import FiltrationOrder, alpha_filtration_order
from .persistence import FieldOps, PersistencePairs, persistent_homology_field
from .simplices import Simplex, SimplicialComplex

R = TypeVar("R")


@dataclass(frozen=True)
class PersistentHomologyResult:
    alpha: AlphaFiltration
    filtration: FiltrationOrder
    persistence: PersistencePairs
    barcode: Barcode


def persistent_homology_from_points(
    points: np.ndarray,
    maximal_simplices: Sequence[Simplex],
    *,
    max_dim: int,
    field: FieldOps[R],
) -> PersistentHomologyResult:
    points_array = np.asarray(points, dtype=float)
    if points_array.ndim != 2:
        raise ValueError("points must have shape (n_points, ambient_dim)")

    alpha = alpha_values_squared(points_array, maximal_simplices, max_dim=max_dim)
    simplices = sorted(alpha.alpha_sq.keys(), key=lambda s: (len(s), s))

    filtration = alpha_filtration_order(alpha, simplices)
    persistence = persistent_homology_field(filtration.simplices, field=field)
    barcode = barcodes_from_persistence(filtration, persistence)

    return PersistentHomologyResult(
        alpha=alpha,
        filtration=filtration,
        persistence=persistence,
        barcode=barcode,
    )