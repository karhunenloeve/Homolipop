from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from homolipop.graph_filtration import proximity_graph_filtration
from homolipop.kgraph import k_theory_cuntz_krieger_Fp
from homolipop.kplotting import KTheoryProfile, plot_k_theory_profile


def main() -> None:
    rng = np.random.default_rng(0)
    points = rng.random((60, 2))

    filtration = proximity_graph_filtration(
        points,
        use_squared_distances=False,
        distance_tolerance=0.0,
        max_steps=80,
    )

    p = 2

    k0_dims = np.empty(filtration.n_steps, dtype=int)
    k1_dims = np.empty(filtration.n_steps, dtype=int)

    for step in range(filtration.n_steps):
        adjacency = filtration.adjacency_matrix(step, include_self_loops=False, dtype=np.int8)
        kt = k_theory_cuntz_krieger_Fp(adjacency, p=p)
        k0_dims[step] = kt.k0_dim
        k1_dims[step] = kt.k1_dim

    profile = KTheoryProfile(
        thresholds=filtration.thresholds,
        k0_dims=k0_dims,
        k1_dims=k1_dims,
        p=p,
    )

    plot_k_theory_profile(profile, title="Homolipop graph C*-K-theory profile over F_2")
    plt.show()


if __name__ == "__main__":
    main()
