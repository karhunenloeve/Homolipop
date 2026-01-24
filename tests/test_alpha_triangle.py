import numpy as np

from robbybubble.alpha import alpha_values_squared


def test_alpha_squared_for_right_triangle_is_half() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    alpha = alpha_values_squared(points, [(0, 1, 2)], max_dim=2)

    assert abs(alpha.alpha_sq[(0, 1, 2)] - 0.5) < 1e-10
    assert alpha.alpha_sq[(0,)] == 0.0
    assert alpha.alpha_sq[(1,)] == 0.0
    assert alpha.alpha_sq[(2,)] == 0.0


def test_alpha_monotonicity_on_triangle_faces() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    tri = (0, 1, 2)
    alpha = alpha_values_squared(points, [tri], max_dim=2)

    tri_value = alpha.alpha_sq[tri]
    for edge in [(0, 1), (0, 2), (1, 2)]:
        assert alpha.alpha_sq[edge] <= tri_value + 1e-12