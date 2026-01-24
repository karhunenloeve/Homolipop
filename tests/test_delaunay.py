import numpy as np

from homolipop import delaunay_d_dim


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

    result = delaunay_d_dim(points_2d)

    assert result.delaunay_simplices
    assert all(len(simplex) == 3 for simplex in result.delaunay_simplices)

    for simplex in result.delaunay_simplices:
        assert len(set(simplex)) == 3
        assert all(0 <= index < len(points_2d) for index in simplex)