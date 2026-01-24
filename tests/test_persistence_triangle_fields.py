# tests/test_persistence_triangle_fields.py
from robbybubble.persistence import field_Fp, persistent_homology_field
from robbybubble.simplices import build_complex


def _triangle_filtration() -> list[tuple[int, ...]]:
    return build_complex([(0, 1, 2)], max_dim=2).all_simplices


def test_triangle_over_F2_has_H1_pair() -> None:
    simplices = _triangle_filtration()
    result = persistent_homology_field(simplices, field=field_Fp(2))

    assert simplices[5] == (1, 2)
    assert simplices[6] == (0, 1, 2)
    assert (5, 6, 1) in set(result.pairs)


def test_triangle_over_F5_has_same_H1_pair() -> None:
    simplices = _triangle_filtration()
    result = persistent_homology_field(simplices, field=field_Fp(5))
    assert (5, 6, 1) in set(result.pairs)