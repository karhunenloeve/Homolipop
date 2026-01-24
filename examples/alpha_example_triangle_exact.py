import numpy as np

from robbybubble.alpha import alpha_values_squared


def main() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )

    delaunay_triangles = [(0, 1, 2)]
    alpha = alpha_values_squared(points, delaunay_triangles, max_dim=2)

    expected_radius_sq = 0.5
    print("triangle alpha^2:", alpha.alpha_sq[(0, 1, 2)])
    print("expected:", expected_radius_sq)

    print("edges:")
    for e in [(0, 1), (0, 2), (1, 2)]:
        print(e, alpha.alpha_sq[e])

    print("vertices:")
    for v in [(0,), (1,), (2,)]:
        print(v, alpha.alpha_sq[v])


if __name__ == "__main__":
    main()