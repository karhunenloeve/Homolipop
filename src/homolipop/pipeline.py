from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, TypeVar

import numpy as np

from .alpha import AlphaFiltration, alpha_values_squared
from .barcodes import Barcode, barcodes_from_persistence
from .filtration import FiltrationOrder, alpha_filtration_order
from .persistence import FieldOps, PersistencePairs, persistent_homology_field
from .simplices import Simplex, SimplicialComplex
from .graph_filtration import GraphFiltration, proximity_graph_filtration
from .kgraph import KTheoryFp, k_theory_cuntz_krieger_Fp
from .kplotting import KTheoryProfile

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




@dataclass(frozen=True)
class KTheoryPipelineResult:
    filtration: GraphFiltration
    profile: KTheoryProfile


def k_theory_profile_from_points(
    points: np.ndarray,
    *,
    p: int,
    use_squared_distances: bool = False,
    distance_tolerance: float = 0.0,
    max_steps: Optional[int] = None,
    include_self_loops: bool = False,
) -> KTheoryPipelineResult:
    filtration = proximity_graph_filtration(
        points,
        use_squared_distances=use_squared_distances,
        distance_tolerance=distance_tolerance,
        max_steps=max_steps,
    )

    n_steps = filtration.n_steps
    k0_dims = np.empty(n_steps, dtype=int)
    k1_dims = np.empty(n_steps, dtype=int)

    for step in range(n_steps):
        adjacency = filtration.adjacency_matrix(step, include_self_loops=include_self_loops, dtype=np.int8)
        kt = k_theory_cuntz_krieger_Fp(adjacency, p=p)
        k0_dims[step] = kt.k0_dim
        k1_dims[step] = kt.k1_dim

    profile = KTheoryProfile(
        thresholds=filtration.thresholds,
        k0_dims=k0_dims,
        k1_dims=k1_dims,
        p=int(p),
    )

    return KTheoryPipelineResult(filtration=filtration, profile=profile)