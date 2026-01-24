import numpy as np

from homolipop import delaunay_d_dim


def main() -> None:
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

    print("Ambient dimension:", points_2d.shape[1])
    print("Number of points:", points_2d.shape[0])
    print("Number of Delaunay simplices:", len(result.delaunay_simplices))
    print("First simplices:", result.delaunay_simplices[:10])


if __name__ == "__main__":
    main()