from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from homolipop.delaunay import delaunay_triangulation
from homolipop.pipeline import persistent_homology_from_points
from homolipop.persistence import field_Fp

from homolipop.persistence_rings import RingStyle, plot_rings_from_barcode, red_palette


@dataclass(frozen=True)
class TwoCirclesParams:
    n_per_circle: int = 180
    radius: float = 1.0
    center_left: Tuple[float, float] = (-1.6, 0.0)
    center_right: Tuple[float, float] = (1.6, 0.0)
    noise_sigma: float = 0.03
    seed: int = 0


def sample_two_circles(p: TwoCirclesParams) -> np.ndarray:
    rng = np.random.default_rng(p.seed)

    theta1 = rng.uniform(0.0, 2.0 * np.pi, size=p.n_per_circle)
    theta2 = rng.uniform(0.0, 2.0 * np.pi, size=p.n_per_circle)

    c1 = np.asarray(p.center_left, dtype=float)
    c2 = np.asarray(p.center_right, dtype=float)

    x1 = c1 + p.radius * np.column_stack((np.cos(theta1), np.sin(theta1)))
    x2 = c2 + p.radius * np.column_stack((np.cos(theta2), np.sin(theta2)))

    x = np.vstack((x1, x2))
    if p.noise_sigma > 0.0:
        x = x + rng.normal(0.0, p.noise_sigma, size=x.shape)

    return x


def main() -> None:
    points = sample_two_circles(TwoCirclesParams())

    # Delaunay triangulation in the ambient space.
    # This function returns a DelaunayResult with attribute `delaunay_simplices`.
    # If your actual attribute name differs, change the next line accordingly.
    del_res = delaunay_triangulation(points)
    maximal_simplices = del_res.delaunay_simplices

    # Persistent homology over F_2
    res = persistent_homology_from_points(
        points,
        maximal_simplices,
        max_dim=2,
        field=field_Fp(2),
    )

    # Beautiful red palette persistence rings for all degrees in the barcode.
    style = RingStyle(
        cmap=red_palette(),
        gap_fraction=0.02,
        min_width_deg=1.0,
        gamma=0.65,
        edgecolor="#2b0000",
        linewidth=0.6,
        alpha=0.95,
        infinite_scale=1.07,
    )

    fig = plot_rings_from_barcode(
        res.barcode,
        ncols=3,
        style=style,
        suptitle="Two circles, alpha filtration, F_2",
    )
    fig.savefig("two_circles_rings.png", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()