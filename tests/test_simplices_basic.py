from homolipop.simplices import canonical_simplex, simplex_dim


def test_canonical_simplex_sorts_and_uniques() -> None:
    assert canonical_simplex([3, 1, 3, 2]) == (1, 2, 3)


def test_simplex_dim() -> None:
    assert simplex_dim((7,)) == 0
    assert simplex_dim((1, 4)) == 1
    assert simplex_dim((0, 2, 5)) == 2
