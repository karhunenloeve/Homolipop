from __future__ import annotations

import numpy as np

from homolipop.delaunay import delaunay_triangulation


def test_delaunay_2d_returns_triangles() -> None:
    points_2d = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.3, 0.6],
        ],
        dtype=float,
    )

    result = delaunay_triangulation(points_2d)

    assert result.delaunay_simplices
    assert all(len(simplex) == 3 for simplex in result.delaunay_simplices)

    n_points = len(points_2d)
    for simplex in result.delaunay_simplices:
        assert len(set(simplex)) == 3
        assert all(0 <= index < n_points for index in simplex)


def test_delaunay_rejects_non_2d_array() -> None:
    points_bad = np.array([0.0, 1.0, 2.0], dtype=float)
    try:
        delaunay_triangulation(points_bad)  # type: ignore[arg-type]
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-2D input array")


def test_delaunay_rejects_too_few_points() -> None:
    points_too_few = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    try:
        delaunay_triangulation(points_too_few)
    except ValueError:
        return
    raise AssertionError("expected ValueError for insufficient points in R^2")