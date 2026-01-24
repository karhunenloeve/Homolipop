# examples/pipeline_toeplitz_k_theory_barcodes_plot.py
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from homolipop.pipeline import persistent_toeplitz_k_theory_from_points
from homolipop.plotting import plot_barcodes


def main() -> None:
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