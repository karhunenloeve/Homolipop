from homolipop.persistence import field_Fp, persistent_homology_field
from homolipop.simplices import build_complex


def test_persistence_single_edge_over_F2() -> None:
    simplices = build_complex([(0, 1)], max_dim=1).all_simplices
    result = persistent_homology_field(simplices, field=field_Fp(2))

    assert simplices == [(0,), (1,), (0, 1)]
    assert (1, 2, 0) in set(result.pairs)
    assert (0, 0) in set(result.unpaired)
