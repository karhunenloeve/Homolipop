"""
Example script: Delaunay triangulation of a small point set.

This file demonstrates :func:`homolipop.delaunay_triangulation` on a planar point cloud.

Mathematical conventions
========================

Point cloud
-----------

Let

.. math::

   X = \\{x_0,\\dots,x_{N-1}\\} \\subset \\mathbb R^d

be a finite set of points, represented by a NumPy array ``points_2d`` of shape ``(N, d)``,
with row ``points_2d[i] = x_i``.

Delaunay triangulation
----------------------

Assume :math:`d \\ge 2`. A Delaunay triangulation of :math:`X` is a simplicial complex
with vertex set :math:`X` whose maximal simplices satisfy the empty circumsphere property.

In the planar case :math:`d=2`, the maximal simplices are triangles.
For a triangle with vertices :math:`x_i, x_j, x_k`, let :math:`B(c,r)` be its circumcircle.
The empty circumsphere property is the condition

.. math::

   B(c,r) \\cap X \\subseteq \\{x_i, x_j, x_k\\}.

Under mild genericity assumptions, the Delaunay triangulation is unique.
If degeneracies occur, there can be multiple valid triangulations; an implementation may
return any one of them.

Representation and output contract
----------------------------------

The function :func:`homolipop.delaunay_triangulation` is assumed to return an object
``result`` with attribute ``delaunay_simplices``.

- ``result.delaunay_simplices`` is an array of shape ``(M, d+1)``.
- Each row is a tuple of distinct vertex indices in ``{0,\\dots,N-1}``.
- Each row represents a maximal simplex of the Delaunay triangulation.
  In this file, :math:`d=2`, so each row represents a triangle.

This script prints basic statistics and the first few maximal simplices.
"""

from __future__ import annotations

import numpy as np

from homolipop import delaunay_triangulation


def main() -> None:
    """
    Compute and print a Delaunay triangulation for a fixed planar point set.

    The script

    1. defines a point cloud in :math:`\\mathbb R^2`
    2. computes a Delaunay triangulation
    3. prints the ambient dimension, number of points, number of maximal simplices, and
       the first few simplices as index tuples
    """
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

    print("Ambient dimension:", points_2d.shape[1])
    print("Number of points:", points_2d.shape[0])
    print("Number of Delaunay simplices:", len(result.delaunay_simplices))
    print("First simplices:", result.delaunay_simplices[:10])


if __name__ == "__main__":
    main()