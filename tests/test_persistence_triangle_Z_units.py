from homolipop.persistence import persistent_homology_unit_ring, ring_Z_units
from homolipop.simplices import build_complex


def test_triangle_over_Z_units_runs_and_pairs_H1() -> None:
    simplices = build_complex([(0, 1, 2)], max_dim=2).all_simplices
    result = persistent_homology_unit_ring(simplices, ring=ring_Z_units())

    assert (5, 6, 1) in set(result.pairs)
