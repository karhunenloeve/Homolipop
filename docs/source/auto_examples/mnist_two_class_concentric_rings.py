"""
Example script: persistent graph homology on MNIST and concentric Betti rings.

This file demonstrates a complete pipeline.

1. Load MNIST from OpenML.
2. Select :math:`n` images of a fixed digit and view them as points in :math:`\\mathbb R^{784}`.
3. Apply PCA to obtain points in :math:`\\mathbb R^{d_\\mathrm{PCA}}`.
4. Build a proximity graph filtration on the reduced point cloud.
5. Compute persistent graph homology with coefficients in :math:`\\mathbb F_p`.
6. Visualize the resulting barcodes for two digit classes as concentric Betti rings.

Mathematical conventions
========================

MNIST as a point cloud
----------------------

Each MNIST image is a grayscale array of size :math:`28\\times 28`.
Flattening yields a vector in :math:`\\mathbb R^{784}`.
After rescaling by :math:`1/255`, each coordinate lies in :math:`[0,1]`.

Fix a digit :math:`d \\in \\{0,\\dots,9\\}` and sample indices
:math:`i_0,\\dots,i_{n-1}` from all images labeled :math:`d`.
This produces a point cloud

.. math::

   X = \\{x_0,\\dots,x_{n-1}\\} \\subset [0,1]^{784} \\subset \\mathbb R^{784}.

PCA reduction
-------------

Let :math:`d_\\mathrm{PCA} \\in \\mathbb N`.
PCA produces a linear map :math:`\\pi : \\mathbb R^{784} \\to \\mathbb R^{d_\\mathrm{PCA}}`
and the reduced point cloud

.. math::

   Z = \\{\\pi(x_0),\\dots,\\pi(x_{n-1})\\} \\subset \\mathbb R^{d_\\mathrm{PCA}}.

This script uses randomized SVD and fixes the random state for reproducibility.

Proximity graph filtration
--------------------------

Let :math:`Z \\subset \\mathbb R^{d_\\mathrm{PCA}}` be finite.
A proximity graph filtration is a nested family of simple graphs

.. math::

   G_0 \\subseteq G_1 \\subseteq \\cdots \\subseteq G_{T-1},

on vertex set :math:`\\{0,\\dots,n-1\\}` such that, for each step :math:`t`,
there exists a threshold :math:`\\varepsilon_t \\ge 0` with the property

.. math::

   \\{i,j\\} \\in E(G_t)
   \\iff
   \\mathrm{dist}(z_i,z_j) \\le \\varepsilon_t,

where ``dist`` is the Euclidean distance if ``use_squared_distances=False``,
and the squared Euclidean distance otherwise.

The object returned by :func:`homolipop.graph_filtration.proximity_graph_filtration`
is assumed to provide

- ``n_steps`` equal to :math:`T`
- a nondecreasing threshold array ``thresholds`` with entries :math:`\\varepsilon_t`
- a method ``adjacency_matrix(t, ...)`` returning the adjacency matrix of :math:`G_t`

Persistent graph homology over :math:`\\mathbb F_p`
---------------------------------------------------

Fix a prime :math:`p` and let :math:`\\mathbb F_p` be the finite field of order :math:`p`.
For each graph :math:`G_t`, define its clique complex

.. math::

   \\mathrm{Cl}(G_t)
   =
   \\{\\sigma \\subseteq V \\mid \\text{all distinct } i,j\\in\\sigma
   \\text{ satisfy } \\{i,j\\}\\in E(G_t)\\}.

Then :math:`\\mathrm{Cl}(G_t)` is an abstract simplicial complex and
:math:`\\mathrm{Cl}(G_t) \\subseteq \\mathrm{Cl}(G_{t+1})`.
The persistent homology groups are

.. math::

   H_k\\bigl(\\mathrm{Cl}(G_t);\\mathbb F_p\\bigr),
   \\qquad k \\ge 0,

with structure maps induced by inclusions.
The routine :func:`homolipop.graph_persistence_fp.persistent_graph_homology_Fp`
is assumed to return barcodes for :math:`k=0` and :math:`k=1`,
represented as finite multisets of intervals :math:`(b,d)` with
:math:`0 \\le b \\le d < \\infty`.

Concentric Betti rings visualization
------------------------------------

The plotting functions consume a mapping ``intervals_by_dim`` of the form

``{0: [(b_1,d_1), ...], 1: [(b_1,d_1), ...], ...}``

and produce a polar ring diagram in which

- the angle parameter discretizes the filtration parameter
- ring thickness and color encode interval multiplicity and persistence length

The exact visual encoding is determined by :class:`homolipop.concentric_betti_rings.BettiRingStyle`.

Numerical and reproducibility notes
-----------------------------------

- OpenML loading depends on network access and the OpenML API.
- The PCA step is randomized but uses a fixed ``random_state``.
- Sampling uses NumPy's ``default_rng(seed)`` and is reproducible for fixed seeds.
- Graph filtration thresholds and persistent homology are deterministic given the reduced points.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

from homolipop.graph_filtration import proximity_graph_filtration
from homolipop.graph_persistence_fp import persistent_graph_homology_Fp

from homolipop.concentric_betti_rings import BettiRingStyle, plot_two_class_betti_rings, red_palette


def mnist_points_for_digit(digit: int, *, n: int = 250, seed: int = 0) -> np.ndarray:
    """
    Return a random subsample of MNIST images of a fixed digit as a point cloud.

    Parameters
    ----------
    digit:
        Digit label :math:`d \\in \\{0,\\dots,9\\}`.
    n:
        Number of images to sample. Sampling is without replacement.
    seed:
        Seed for the NumPy random generator used to select indices.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n, 784)`` with entries in :math:`[0,1]`.
        Row ``i`` is the flattened and rescaled image vector.

    Raises
    ------
    ValueError
        If ``n`` exceeds the number of available images for the requested digit.

    Notes
    -----
    The dataset is fetched via OpenML. This requires network access at runtime.
    """
    rng = np.random.default_rng(seed)
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(int)

    idx = np.flatnonzero(y == int(digit))
    sel = rng.choice(idx, size=n, replace=False)
    return X[sel]


def barcode_for_points(points: np.ndarray, *, p: int = 2, pca_dim: int = 30, max_steps: int = 80):
    """
    Compute persistent graph homology barcodes for a point cloud.

    The pipeline is

    .. math::

       \\text{points} \\xrightarrow{\\mathrm{PCA}} Z
       \\xrightarrow{\\text{proximity filtration}} (G_t)_{t=0}^{T-1}
       \\xrightarrow{\\mathrm{Cl}} (\\mathrm{Cl}(G_t))_{t=0}^{T-1}
       \\xrightarrow{H_*} \\text{barcodes}.

    Parameters
    ----------
    points:
        Array of shape ``(N, D)`` representing :math:`N` points in :math:`\\mathbb R^D`.
    p:
        Prime modulus defining the coefficient field :math:`\\mathbb F_p`.
        The implementation is assumed to interpret this as arithmetic modulo ``p``.
    pca_dim:
        Target PCA dimension :math:`d_\\mathrm{PCA}`.
        Must satisfy ``1 <= pca_dim <= D``.
    max_steps:
        Maximum number of filtration steps :math:`T`.

    Returns
    -------
    dict
        Dictionary ``intervals_by_dim`` mapping dimensions to lists of persistence intervals.
        This function merges the output for :math:`H_0` and :math:`H_1` into a single mapping.

    Notes
    -----
    This function assumes that :func:`persistent_graph_homology_Fp` returns an object ``res``
    whose attributes ``h0`` and ``h1`` each provide ``intervals_by_dim``.
    The returned intervals use the threshold values ``filt.thresholds`` as filtration parameters.

    The filtration is built using Euclidean distances, because ``use_squared_distances=False``.
    """
    Z = PCA(n_components=pca_dim, random_state=0, svd_solver="randomized").fit_transform(points)

    filt = proximity_graph_filtration(Z, use_squared_distances=False, distance_tolerance=0.0, max_steps=max_steps)
    adj_by_step = [filt.adjacency_matrix(s, include_self_loops=False, dtype=np.int8) for s in range(filt.n_steps)]

    res = persistent_graph_homology_Fp(adj_by_step, thresholds=filt.thresholds, p=p)

    intervals_by_dim = {}
    intervals_by_dim.update(res.h0.intervals_by_dim)
    intervals_by_dim.update(res.h1.intervals_by_dim)
    return intervals_by_dim


def main() -> None:
    """
    Compute and plot concentric Betti rings for two MNIST digit classes.

    The script

    - computes barcodes in dimensions 0 and 1 for digit 3 and digit 8
    - renders a two-class concentric ring plot
    - saves the plot as a PNG file

    Output
    ------
    The file ``mnist_3_vs_8_concentric_betti_rings.png`` is written to the working directory.
    """
    ints_A = barcode_for_points(mnist_points_for_digit(3, n=250, seed=0), p=2, pca_dim=30, max_steps=80)
    ints_B = barcode_for_points(mnist_points_for_digit(8, n=250, seed=1), p=2, pca_dim=30, max_steps=80)

    style = BettiRingStyle(
        cmap=red_palette(),
        bins=720,
        base_ring_width=1.0,
        ring_gap=0.10,
        linewidth=0.22,
        alpha=0.98,
    )

    fig = plot_two_class_betti_rings(
        ints_A,
        ints_B,
        label_a="MNIST digit 3",
        label_b="MNIST digit 8",
        dims=(0, 1),
        style=style,
        figsize=(12, 6),
    )
    fig.savefig("mnist_3_vs_8_concentric_betti_rings.png", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()