"""
Toeplitz graph model: barcodes for finite field K-theory presentations.

This example demonstrates :func:`homolipop.pipeline.persistent_toeplitz_k_theory_from_points`,
a graph based pipeline motivated by Toeplitz graph C star algebras.

The output consists of two barcode plots that represent persistence modules over
:math:`\\mathbb F_2` associated to presentation data for :math:`K_0` and :math:`K_1`.

A crucial point
===============

This pipeline does not compute persistent :math:`K`-theory as a functor on C star algebras.

Instead, at each filtration step it forms finite field linear invariants derived from the
standard presentation matrices appearing in graph C star algebra K-theory formulas and then
computes persistence for the induced linear maps.

Pipeline
========

1. Sample a finite point cloud :math:`X \\subset \\mathbb R^2`.
2. Build a deterministic fixed vertex graph filtration on :math:`V = \\{0,\\dots,n-1\\}`.
3. From each graph step, form an adjacency matrix :math:`A_s`.
4. Over :math:`\\mathbb F_p`, consider the linear map

   .. math::

      \\phi_s = I - A_s^{\\mathsf T} : \\mathbb F_p^n \\to \\mathbb F_p^n.

5. Record the finite field dimensions of

   .. math::

      \\operatorname{coker}(\\phi_s),
      \\qquad
      \\ker(\\phi_s).

6. Compute persistence of these assignments along the filtration and display barcodes.

Mathematical meaning
====================

For a 0--1 matrix :math:`A`, the Cuntz--Krieger and Toeplitz graph algebra frameworks yield
presentation formulas for integral :math:`K`-groups involving the abelian groups

.. math::

   \\operatorname{coker}(I - A^{\\mathsf T}),
   \\qquad
   \\ker(I - A^{\\mathsf T}).

This example computes the corresponding objects after base change to :math:`\\mathbb F_2`.
The resulting objects are finite dimensional :math:`\\mathbb F_2` vector spaces, and the
filtration maps are induced by the graph quotient direction conventions implemented in
:func:`homolipop.pipeline.persistent_toeplitz_k_theory_from_points`.

Parameters
==========

- ``p`` is the prime defining :math:`\\mathbb F_p`.
- ``neighbor_rank`` selects a sparse proximity rule for the graph filtration.
  The precise rule is defined by the implementation of the pipeline function.
- ``use_squared_distances`` and ``distance_tolerance`` control numerical details of the
  thresholding.
- ``include_self_loops`` determines whether diagonal entries in the adjacency matrices are
  allowed.

Reproducibility
===============

All randomness in this file is controlled by NumPy's ``default_rng`` with a fixed seed.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from homolipop.pipeline import persistent_toeplitz_k_theory_from_points
from homolipop.plotting import plot_barcodes


def main() -> None:
    """
    Compute and plot Toeplitz graph K0 and K1 presentation barcodes over :math:`\\mathbb F_2`.

    The script opens two barcode plots and displays them in a Matplotlib window.
    """
    rng = np.random.default_rng(0)
    points = rng.random((60, 2))

    result = persistent_toeplitz_k_theory_from_points(
        points,
        p=2,
        neighbor_rank=1,
        use_squared_distances=False,
        distance_tolerance=0.0,
        include_self_loops=False,
    )

    plot_barcodes(result.h0, title="Toeplitz graph K0 ⊗ F_2 barcode", figsize=(10.0, 4.0))
    plot_barcodes(result.h1, title="Toeplitz graph K1 ⊗ F_2 barcode", figsize=(10.0, 4.0))

    plt.show()


if __name__ == "__main__":
    main()