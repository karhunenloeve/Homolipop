from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .edge_filtration import EdgeFiltration, directed_adjacency_from_undirected, edge_filtration_fixed_vertices
from .k0_like import K0LikePersistenceResultFp, k0_like_persistence_quotient_direction_Fp


@dataclass(frozen=True)
class K0LikePipelineResultFp:
    p: int
    thresholds: np.ndarray
    k0_like: K0LikePersistenceResultFp
    vertex_order: np.ndarray
    edge_filtration: EdgeFiltration


def k0_like_persistence_from_points_Fp(
    points: np.ndarray,
    *,
    p: int,
    thresholds: Optional[np.ndarray] = None,
    n_steps: int = 50,
    metric: str = "euclidean",
    tolerance: float = 0.0,
    include_self_loops: bool = False,
    deterministic_order: bool = True,
    orientation: str = "lower_to_higher",
    include_both_directions: bool = False,
) -> K0LikePipelineResultFp:
    filt = edge_filtration_fixed_vertices(
        points,
        thresholds=thresholds,
        n_steps=n_steps,
        metric=metric,  # type: ignore[arg-type]
        tolerance=tolerance,
        include_self_loops=include_self_loops,
        deterministic_order=deterministic_order,
    )

    undirected_by_step = filt.adjacency_matrices()

    directed_by_step = [
        directed_adjacency_from_undirected(
            a,
            orientation=orientation,  # type: ignore[arg-type]
            include_both_directions=include_both_directions,
        )
        for a in undirected_by_step
    ]

    k0_like = k0_like_persistence_quotient_direction_Fp(
        directed_by_step,
        thresholds=filt.thresholds,
        p=p,
    )

    return K0LikePipelineResultFp(
        p=int(p),
        thresholds=filt.thresholds,
        k0_like=k0_like,
        vertex_order=filt.order,
        edge_filtration=filt,
    )


__all__ = [
    "K0LikePipelineResultFp",
    "k0_like_persistence_from_points_Fp",
]
