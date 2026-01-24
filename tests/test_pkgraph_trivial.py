from __future__ import annotations

import numpy as np

from homolipop.pkgraph import persistent_kgraph_from_nested_matrices


def test_identity_matrices_give_no_h1_and_no_finite_h0_deaths() -> None:
    matrices = [np.eye(k, dtype=int) for k in range(1, 6)]
    result = persistent_kgraph_from_nested_matrices(matrices, p=2)

    assert result.h1.intervals_by_dim.get(1, []) == []

    h0 = result.h0.intervals_by_dim.get(0, [])
    finite = [(b, d) for (b, d) in h0 if d is not None]
    assert finite == []