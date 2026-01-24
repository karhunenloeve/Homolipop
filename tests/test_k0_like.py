from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from homolipop.graph_persistence_fp import persistent_graph_homology_Fp
from homolipop.k0_like import (
    k0_like_persistence_from_quotient_system_Fp,
    k0_like_persistence_quotient_direction_Fp,
)

Interval = Tuple[float, Optional[float]]


def _intervals(barcode, dim: int) -> List[Interval]:
    return list(barcode.intervals_by_dim.get(dim, []))


def _sorted_intervals(intervals: List[Interval]) -> List[Interval]:
    return sorted(intervals, key=lambda x: (float(x[0]), float("inf") if x[1] is None else float(x[1])))


def _adj(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    a = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        a[int(u), int(v)] = 1
    return a


def test_empty_graph_has_n_infinite_H0_intervals() -> None:
    n = 4
    steps = 3
    thresholds = [0.0, 1.0, 2.0]
    directed = [_adj(n, []) for _ in range(steps)]

    res = k0_like_persistence_from_quotient_system_Fp(directed, thresholds=thresholds, p=2)

    h0 = _sorted_intervals(_intervals(res.k0_like, 0))
    assert len(h0) == n
    assert all(death is None for _, death in h0)
    assert all(birth == thresholds[0] for birth, _ in h0)


def test_tree_edges_kill_components_one_by_one() -> None:
    n = 4
    thresholds = [0.0, 1.0, 2.0, 3.0]

    directed = [
        _adj(n, []),
        _adj(n, [(0, 1)]),
        _adj(n, [(0, 1), (1, 2)]),
        _adj(n, [(0, 1), (1, 2), (2, 3)]),
    ]

    res = k0_like_persistence_from_quotient_system_Fp(directed, thresholds=thresholds, p=2)
    h0 = _sorted_intervals(_intervals(res.k0_like, 0))

    assert len(h0) == n
    infinite = [iv for iv in h0 if iv[1] is None]
    finite = [iv for iv in h0 if iv[1] is not None]

    assert len(infinite) == 1
    assert len(finite) == n - 1

    finite_deaths = sorted(float(d) for _, d in finite if d is not None)
    assert finite_deaths == [1.0, 2.0, 3.0]


def test_connected_final_graph_has_one_infinite_H0_interval() -> None:
    n = 5
    thresholds = [0.0, 1.0]
    directed = [
        _adj(n, []),
        _adj(n, [(0, 1), (1, 2), (2, 3), (3, 4)]),
    ]

    res = k0_like_persistence_from_quotient_system_Fp(directed, thresholds=thresholds, p=2)
    h0 = _sorted_intervals(_intervals(res.k0_like, 0))

    infinite = [iv for iv in h0 if iv[1] is None]
    assert len(infinite) == 1


def test_quotient_direction_matches_reversed_filtration_computation() -> None:
    n = 4
    thresholds = [0.0, 1.0, 2.0]
    directed = [
        _adj(n, []),
        _adj(n, [(0, 1)]),
        _adj(n, [(0, 1), (2, 3)]),
    ]

    res_q = k0_like_persistence_quotient_direction_Fp(directed, thresholds=thresholds, p=2)

    rev_directed = list(reversed(directed))
    rev_thresholds = list(reversed(thresholds))
    rev_graph = persistent_graph_homology_Fp(rev_directed, thresholds=rev_thresholds, p=2)

    q_h0 = _sorted_intervals(_intervals(res_q.k0_like, 0))
    rev_h0 = _sorted_intervals(_intervals(rev_graph.h0, 0))
    assert q_h0 == rev_h0