from robbybubble.simplices import build_complex


def test_build_complex_closure_for_two_triangles() -> None:
    maximal = [(0, 1, 3), (1, 2, 3)]
    complex_data = build_complex(maximal, max_dim=2)

    expected_vertices = [(0,), (1,), (2,), (3,)]
    expected_edges = [(0, 1), (0, 3), (1, 2), (1, 3), (2, 3)]
    expected_triangles = [(0, 1, 3), (1, 2, 3)]

    assert complex_data.simplices_by_dim[0] == expected_vertices
    assert complex_data.simplices_by_dim[1] == expected_edges
    assert complex_data.simplices_by_dim[2] == expected_triangles

    for simplex in expected_vertices + expected_edges + expected_triangles:
        assert simplex in complex_data.index
        assert complex_data.all_simplices[complex_data.index[simplex]] == simplex