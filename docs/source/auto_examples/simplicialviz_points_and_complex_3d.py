"""
Visualizing a Delaunay 2-skeleton in :math:`\\mathbb R^3`.

This example samples random points in :math:`\\mathbb R^3`, computes a Delaunay
triangulation, builds the simplicial complex given by the downward closure of the
maximal Delaunay simplices, truncates to the 2-skeleton, and visualizes it using
:mod:`simplicialviz`.

Pipeline
========

1. Sample a finite point cloud :math:`X \\subset \\mathbb R^3`.
2. Compute a Delaunay triangulation of :math:`X`.
3. Form the simplicial complex consisting of all faces of the maximal Delaunay simplices.
4. Extract the 2-skeleton, meaning all simplices of dimension at most 2.
5. Plot vertices, edges, and triangular faces in 3D.

Mathematical conventions
========================

Delaunay complex
----------------

Let :math:`X = \\{x_0,\\dots,x_{n-1}\\} \\subset \\mathbb R^3` be in general position.
A Delaunay triangulation yields a simplicial complex whose maximal simplices are
3-simplices. The 2-skeleton of this complex is the subcomplex consisting of all
simplices of dimensions 0, 1, and 2.

Simplicial closure
------------------

Given a list of maximal simplices, :func:`homolipop.simplices.build_complex` is assumed
to return the downward closure, truncated to simplices of dimension at most ``max_dim``.
In particular, every face of every maximal simplex is included.

Visualization
=============

The function :func:`simplicialviz.plot_complex_3d` renders the point cloud together
with edges and triangular faces. The style is controlled by :class:`simplicialviz.Plot3DStyle`.

Reproducibility
===============

All randomness in this file is controlled by NumPy's ``default_rng`` with a fixed seed.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from homolipop.delaunay import delaunay_triangulation
from homolipop.simplices import build_complex
from simplicialviz import Plot3DStyle, plot_complex_3d


def main() -> None:
    """
    Compute and plot the Delaunay 2-skeleton of random points in :math:`\\mathbb R^3`.

    The script opens an interactive Matplotlib window and closes the figure after display.
    """
    rng = np.random.default_rng(0)
    points = rng.random((60, 3))

    delaunay = delaunay_triangulation(points)
    complex_data = build_complex(delaunay.delaunay_simplices, max_dim=2)

    style = Plot3DStyle(alpha_faces=0.12, line_width=0.5)
    fig = plot_complex_3d(points, complex_data)
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()