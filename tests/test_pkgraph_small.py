from __future__ import annotations

import numpy as np

from homolipop.pkgraph import persistent_kgraph_from_nested_matrices


def test_small_nested_matrix_produces_h1_interval() -> None:
    p = 2

    m1 = np.array([[0]], dtype=int)
    m2 = np.array([[0, 0], [0, 0]], dtype=int)
    m3 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=int)

    result = persistent_kgraph_from_nested_matrices([m1, m2, m3], p=p)

    h1 = result.h1.intervals_by_dim.get(1, [])
    assert len(h1) >= 1
    assert all(d is None for (_, d) in h1)
