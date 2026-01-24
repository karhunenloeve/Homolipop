import numpy as np

from homolipop.alpha import alpha_values_squared
from homolipop.delaunay import delaunay_d_dim
from homolipop.simplices import build_complex, iter_faces


def test_alpha_defined_for_all_simplices_in_closure() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.3, 0.6],
        ],
        dtype=float,
    )

    delaunay = delaunay_d_dim(points)
    triangles = delaunay.delaunay_simplices

    max_dim = 2
    complex_data = build_complex(triangles, max_dim=max_dim)
    alpha = alpha_values_squared(points, triangles, max_dim=max_dim)

    for simplex in complex_data.all_simplices:
        assert simplex in alpha.alpha_sq


def test_alpha_monotonicity_on_all_incidences() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.3, 0.6],
        ],
        dtype=float,
    )

    delaunay = delaunay_d_dim(points)
    triangles = delaunay.delaunay_simplices

    max_dim = 2
    complex_data = build_complex(triangles, max_dim=max_dim)
    alpha = alpha_values_squared(points, triangles, max_dim=max_dim)

    for simplex in complex_data.all_simplices:
        dim = len(simplex) - 1
        if dim <= 0:
            continue

        value = alpha.alpha_sq[simplex]
        for face in iter_faces(simplex, face_dim=dim - 1):
            assert alpha.alpha_sq[face] <= value + 1e-12