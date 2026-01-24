from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from homolipop.delaunay import delaunay_triangulation
from homolipop.simplices import build_complex
from simplicialviz import Plot3DStyle, plot_complex_3d


def main() -> None:
    rng = np.random.default_rng(0)
    points = rng.random((60, 3))

    delaunay = delaunay_triangulation(points)
    complex_data = build_complex(delaunay.delaunay_simplices, max_dim=2)

    style = Plot3DStyle(title="Delaunay 2-skeleton in R^3", face_alpha=0.12, edge_alpha=0.5)
    fig = plot_complex_3d(points, complex_data, max_dim=2, style=style)
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()