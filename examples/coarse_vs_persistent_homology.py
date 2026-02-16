"""
Compare persistent homology with a computable proxy for bornological coarse homology.

Interpretation
--------------
For a metric space X, a standard computable proxy for coarse homology is the direct limit
over Rips complexes as the scale parameter r -> infinity. On a bounded finite point cloud,
this stabilizes to the homology of a point. In contrast, persistent homology detects
mesoscopic features that live for a range of scales but eventually die.

This script:
1. Samples a noisy circle point cloud in R^2.
2. Computes Vietoris–Rips persistent homology over F_2 up to dimension 1.
3. Defines "coarse homology at scale R" as the classes whose persistence intervals
   reach the terminal scale R (i.e. do not die before R).
4. Plots barcodes and prints the coarse Betti numbers at the terminal scale.

No files are written. Plots are shown interactively.

Requirements
------------
- homolipop.persistence.field_Fp
- homolipop.persistence.persistent_homology_field
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import inf
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from homolipop.persistence import field_Fp, persistent_homology_field

Simplex = tuple[int, ...]
Interval = Tuple[float, float]


@dataclass(frozen=True, slots=True)
class RipsPHResult:
    simplices: List[Simplex]
    filtration: np.ndarray
    intervals_by_dim: Dict[int, List[Interval]]


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array of shape (n, d)")
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _rips_filtration_simplices(
    dist: np.ndarray, *, max_dim: int, max_radius: float
) -> List[Tuple[float, Simplex]]:
    """
    Build Vietoris–Rips simplices up to max_dim with filtration value = max edge length.

    Returned as (filtration_value, simplex) pairs.
    """
    n = dist.shape[0]
    if dist.shape != (n, n):
        raise ValueError("dist must be a square matrix")

    out: List[Tuple[float, Simplex]] = []

    # 0-simplices at filtration 0
    for i in range(n):
        out.append((0.0, (i,)))

    if max_dim >= 1:
        for i, j in combinations(range(n), 2):
            w = float(dist[i, j])
            if w <= max_radius:
                out.append((w, (i, j)))

    if max_dim >= 2:
        for i, j, k in combinations(range(n), 3):
            w = float(max(dist[i, j], dist[i, k], dist[j, k]))
            if w <= max_radius:
                out.append((w, (i, j, k)))

    # Ensure faces come before cofaces:
    # primary key: filtration
    # secondary: dimension
    # tertiary: lexicographic simplex
    out.sort(key=lambda t: (t[0], len(t[1]), t[1]))
    return out


def compute_rips_persistent_homology(
    points: np.ndarray, *, max_dim: int, max_radius: float
) -> RipsPHResult:
    dist = _pairwise_distances(points)
    fs = _rips_filtration_simplices(dist, max_dim=max_dim, max_radius=max_radius)

    filtration = np.array([f for (f, _) in fs], dtype=float)
    simplices = [s for (_, s) in fs]

    pairs = persistent_homology_field(simplices, field=field_Fp(2))

    # Convert index pairs into (birth, death) intervals per homological dimension.
    intervals_by_dim: Dict[int, List[Interval]] = {d: [] for d in range(max_dim + 1)}

    for dim, i_birth, i_death in pairs.pairs:
        b = float(filtration[i_birth])
        d = float(filtration[i_death])
        if dim <= max_dim:
            intervals_by_dim[dim].append((b, d))

    for dim, i_birth in pairs.unpaired:
        b = float(filtration[i_birth])
        if dim <= max_dim:
            intervals_by_dim[dim].append((b, inf))

    for dim in intervals_by_dim:
        intervals_by_dim[dim].sort(key=lambda t: (t[0], t[1]))

    return RipsPHResult(
        simplices=simplices,
        filtration=filtration,
        intervals_by_dim=intervals_by_dim,
    )


def coarse_betti_at_scale(intervals: Iterable[Interval], *, scale: float) -> int:
    """
    Coarse proxy: count classes surviving up to the terminal scale.
    """
    s = float(scale)
    c = 0
    for b, d in intervals:
        if b <= s and (d == inf or d >= s):
            c += 1
    return c


def plot_barcode(ax: plt.Axes, intervals: List[Interval], *, scale: float, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("scale")
    ax.set_yticks([])

    ax.axvline(scale, linestyle="--")

    if not intervals:
        ax.text(0.5, 0.5, "no intervals", transform=ax.transAxes, ha="center", va="center")
        ax.set_xlim(0.0, scale * 1.05)
        ax.set_ylim(-1, 1)
        return

    ys = np.arange(len(intervals), dtype=float)
    for y, (b, d) in zip(ys, intervals):
        right = scale if d == inf else d
        ax.hlines(y, b, right, linewidth=2)
        if d != inf:
            ax.plot([d], [y], marker="o", markersize=3)

    xmax = max(scale, max((d if d != inf else scale) for _, d in intervals)) * 1.05
    ax.set_ylim(-1, len(intervals))
    ax.set_xlim(0.0, xmax)


def sample_noisy_circle(*, n: int, radius: float, noise: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    pts = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    pts += rng.normal(scale=noise, size=pts.shape)
    return pts


def load_digits_points_2d(*, digits: tuple[int, ...], n_per_digit: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Real dataset, offline: sklearn.datasets.load_digits.

    Returns
    -------
    points : (N, 2) float array
        PCA projection of digit images to R^2.
    labels : (N,) int array
        Digit labels for coloring.
    """
    try:
        from sklearn.datasets import load_digits
        from sklearn.decomposition import PCA
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "This example needs scikit-learn. Install via `pip install scikit-learn`."
        ) from e

    rng = np.random.default_rng(seed)
    ds = load_digits()
    X = ds.data.astype(float)
    y = ds.target.astype(int)

    idx_all: list[int] = []
    for d in digits:
        idx = np.flatnonzero(y == d)
        if len(idx) < n_per_digit:
            raise ValueError(f"not enough samples for digit {d}: have {len(idx)}, need {n_per_digit}")
        take = rng.choice(idx, size=n_per_digit, replace=False)
        idx_all.extend(map(int, take))

    idx_all = np.array(idx_all, dtype=int)
    X_sel = X[idx_all]
    y_sel = y[idx_all]

    # Standardize features lightly: center only (scale optional, but keep dependencies minimal).
    X_sel = X_sel - X_sel.mean(axis=0, keepdims=True)

    pts = PCA(n_components=2, random_state=seed).fit_transform(X_sel)
    return pts, y_sel


def _intervals_surviving_to_scale(intervals: list[Interval], *, scale: float) -> list[Interval]:
    s = float(scale)
    return [(b, d) for (b, d) in intervals if b <= s and (d == inf or d >= s)]


def main() -> None:
    # Real dataset: digits 0 and 8 often yield a more structured point cloud than a circle.
    points, labels = load_digits_points_2d(digits=(0, 8), n_per_digit=80, seed=0)

    max_dim = 1
    # Choose a terminal scale from the data, so the example is stable under PCA scaling.
    dist = _pairwise_distances(points)
    # A “large” scale: 80th percentile of distances (tunable, but robust).
    max_radius = float(np.quantile(dist, 0.80))

    res = compute_rips_persistent_homology(points, max_dim=max_dim, max_radius=max_radius)

    h0 = res.intervals_by_dim[0]
    h1 = res.intervals_by_dim[1]
    h0_coarse = _intervals_surviving_to_scale(h0, scale=max_radius)
    h1_coarse = _intervals_surviving_to_scale(h1, scale=max_radius)

    print("Terminal scale R = {:.6f}".format(max_radius))
    print("Coarse proxy Betti at R:")
    print("beta_0^coarse(R) =", len(h0_coarse))
    print("beta_1^coarse(R) =", len(h1_coarse))
    print()
    print("Interpretation:")
    print("- Persistent features can be long-lived but still die before the terminal scale.")
    print("- The coarse proxy keeps only classes surviving to scale R.")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # Point cloud
    ax = axes[0, 0]
    ax.set_title("Digits (0 and 8), PCA to R^2")
    sc = ax.scatter(points[:, 0], points[:, 1], s=12, c=labels)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    # Normal persistence barcodes
    plot_barcode(axes[0, 1], h0, scale=max_radius, title="Persistent barcode H0 (all intervals)")
    plot_barcode(axes[0, 2], h1, scale=max_radius, title="Persistent barcode H1 (all intervals)")

    # Coarse proxy = surviving intervals only, plotted “next to” persistent
    plot_barcode(axes[1, 1], h0_coarse, scale=max_radius, title="Coarse proxy at R: surviving H0")
    plot_barcode(axes[1, 2], h1_coarse, scale=max_radius, title="Coarse proxy at R: surviving H1")

    # Text panel
    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.0,
        1.0,
        "\n".join(
            [
                "Coarse proxy definition",
                "",
                "At terminal scale R:",
                "keep intervals [b,d) with b <= R <= d (or d = +inf).",
                "",
                "This mimics the direct-limit intuition:",
                "classes must survive to arbitrarily large scales,",
                "but on bounded data the proxy tends to kill H1.",
            ]
        ),
        va="top",
        ha="left",
    )

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()