# tests/test_vertex_growth_filtration_nested.py
from __future__ import annotations

import numpy as np

from homolipop.vertex_growth_filtration import vertex_growth_filtration


def test_adjacency_is_nested_principal_submatrices() -> None:
    rng = np.random.default_rng(0)
    points = rng.random((25, 3))

    filt = vertex_growth_filtration(points, neighbor_rank=2)
    a = filt.adjacency

    n = a.shape[0]
    for k in range(1, n):
        assert np.array_equal(a[:k, :k], a[: k + 1, : k + 1][:k, :k])


def test_step_values_monotone() -> None:
    rng = np.random.default_rng(1)
    points = rng.random((30, 2))

    filt = vertex_growth_filtration(points, neighbor_rank=3)
    values = filt.step_values

    assert values.shape == (points.shape[0],)
    assert np.all(values[1:] >= values[:-1])
