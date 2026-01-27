import numpy as np

from homolipop.alpha import alpha_values_squared
from homolipop.delaunay import delaunay_triangulation
from homolipop.simplices import build_complex


def main():
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

    delaunay = delaunay_triangulation(points)
    triangles = delaunay.delaunay_simplices

    max_dim = 2
    complex_data = build_complex(triangles, max_dim=max_dim)
    alpha = alpha_values_squared(points, triangles, max_dim=max_dim)

    print("Number of points:", len(points))
    print("Number of Delaunay triangles:", len(triangles))
    print("Number of simplices up to dim 2:", len(complex_data.all_simplices))

    print("\nSome simplices and alpha^2 values:")
    for simplex in complex_data.all_simplices[:12]:
        print(simplex, "->", alpha.alpha_sq[simplex])

    print("\nMonotonicity checks (alpha(face) <= alpha(simplex)) on first 10 triangles:")
    for tri in triangles[:10]:
        a_tri = alpha.alpha_sq[tri]
        edges = [(tri[0], tri[1]), (tri[0], tri[2]), (tri[1], tri[2])]
        for e in edges:
            if alpha.alpha_sq[e] > a_tri:
                raise RuntimeError(f"monotonicity violated: {e} has {alpha.alpha_sq[e]} > {a_tri} of {tri}")

    print("\nOK")


if __name__ == "__main__":
    main()
