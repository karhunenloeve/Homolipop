from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

from homolipop.graph_filtration import proximity_graph_filtration
from homolipop.graph_persistence_fp import persistent_graph_homology_Fp

from homolipop.concentric_betti_rings import BettiRingStyle, plot_two_class_betti_rings, red_palette


def mnist_points_for_digit(digit: int, *, n: int = 250, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(int)

    idx = np.flatnonzero(y == int(digit))
    sel = rng.choice(idx, size=n, replace=False)
    return X[sel]


def barcode_for_points(points: np.ndarray, *, p: int = 2, pca_dim: int = 30, max_steps: int = 80):
    Z = PCA(n_components=pca_dim, random_state=0, svd_solver="randomized").fit_transform(points)

    filt = proximity_graph_filtration(Z, use_squared_distances=False, distance_tolerance=0.0, max_steps=max_steps)
    adj_by_step = [filt.adjacency_matrix(s, include_self_loops=False, dtype=np.int8) for s in range(filt.n_steps)]

    res = persistent_graph_homology_Fp(adj_by_step, thresholds=filt.thresholds, p=p)

    intervals_by_dim = {}
    intervals_by_dim.update(res.h0.intervals_by_dim)
    intervals_by_dim.update(res.h1.intervals_by_dim)
    return intervals_by_dim


def main() -> None:
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