from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .barcodes import Barcode
from .graph_persistence_fp import GraphPersistenceResult, persistent_graph_homology_Fp

Edge = Tuple[int, int]


@dataclass(frozen=True)
class K0LikePersistenceResultFp:
    p: int
    thresholds: np.ndarray
    k0_like: Barcode


def k0_like_persistence_from_quotient_system_Fp(
    directed_adjacency_by_step: Sequence[np.ndarray],
    *,
    thresholds: Sequence[float],
    p: int,
) -> K0LikePersistenceResultFp:
    result = persistent_graph_homology_Fp(
        directed_adjacency_by_step,
        thresholds=thresholds,
        p=p,
    )
    return K0LikePersistenceResultFp(p=result.p, thresholds=result.thresholds, k0_like=result.h0)


def k0_like_persistence_quotient_direction_Fp(
    directed_adjacency_by_step: Sequence[np.ndarray],
    *,
    thresholds: Sequence[float],
    p: int,
) -> K0LikePersistenceResultFp:
    thresholds_arr = np.asarray(thresholds, dtype=float)
    if thresholds_arr.ndim != 1:
        raise ValueError("thresholds must be 1D")
    if len(directed_adjacency_by_step) != int(thresholds_arr.size):
        raise ValueError("directed_adjacency_by_step and thresholds must have same length")

    if not directed_adjacency_by_step:
        return K0LikePersistenceResultFp(p=int(p), thresholds=thresholds_arr, k0_like=Barcode(intervals_by_dim={}))

    rev_adj = list(reversed(directed_adjacency_by_step))
    rev_thr = thresholds_arr[::-1].copy()

    rev_result = persistent_graph_homology_Fp(rev_adj, thresholds=rev_thr, p=p)
    return K0LikePersistenceResultFp(p=rev_result.p, thresholds=thresholds_arr, k0_like=rev_result.h0)


def k0_like_interpretation() -> Dict[str, str]:
    return {
        "K0_like(t)": "F_p^V / span{ e_v - e_u : u->v is an edge in E_t }",
        "identification": "K0_like(t) is canonically isomorphic to H_0(E_t; F_p) for the directed 1-skeleton chain model",
        "covariant_persistence": "for edge inclusions E_s ⊆ E_t one gets maps K0_like(s) → K0_like(t) induced by identity on vertices",
        "quotient_direction": "for Toeplitz quotient maps A_t → A_s with s ≤ t, the induced map corresponds to K0_like(t) → K0_like(s); this is handled by reversing the filtration order",
    }


__all__ = [
    "K0LikePersistenceResultFp",
    "k0_like_persistence_from_quotient_system_Fp",
    "k0_like_persistence_quotient_direction_Fp",
    "k0_like_interpretation",
]
