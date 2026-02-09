from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from homolipop.pipeline_k0_like import k0_like_persistence_from_points_Fp
from homolipop.plotting import plot_barcodes


def main() -> None:
    rng = np.random.default_rng(0)
    points = rng.random((80, 2))

    result = k0_like_persistence_from_points_Fp(
        points,
        p=2,
        n_steps=40,
        deterministic_order=True,
        orientation="lower_to_higher",
        include_both_directions=False,
    )

    fig = plot_barcodes(
        result.k0_like.k0_like,
        title="K0-like persistence over F_2 in Toeplitz quotient direction",
    )
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
