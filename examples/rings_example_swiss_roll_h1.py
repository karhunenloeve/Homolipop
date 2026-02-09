from __future__ import annotations

import numpy as np

from homolipop.delaunay import delaunay_triangulation
from homolipop.pipeline import persistent_homology_from_points
from homolipop.persistence import field_Fp
from homolipop.persistence_rings import RingStyle, plot_rings_from_barcode, red_palette


def sample_swiss_roll(n: int = 600, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.random(n))
    x = t * np.cos(t)
    y = 21.0 * rng.random(n)
    z = t * np.sin(t)
    pts = np.column_stack((x, z))  # project to R^2 to get a loop-like structure
    pts = pts + rng.normal(0.0, 0.15, size=pts.shape)
    return pts


def main() -> None:
    points = sample_swiss_roll()

    del_res = delaunay_triangulation(points)
    maximal_simplices = del_res.delaunay_simplices

    res = persistent_homology_from_points(
        points,
        maximal_simplices,
        max_dim=2,
        field=field_Fp(2),
    )

    style = RingStyle(cmap=red_palette(), min_width_deg=1.0, gamma=0.7)

    fig = plot_rings_from_barcode(
        res.barcode,
        ncols=3,
        style=style,
        suptitle="Swiss-roll projection, alpha filtration, F_2",
    )
    fig.savefig("swiss_roll_rings.png", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()