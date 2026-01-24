from homolipop.simplices import iter_faces


def test_iter_faces_edges_of_triangle() -> None:
    triangle = (0, 2, 5)
    edges = list(iter_faces(triangle, face_dim=1))
    assert edges == [(0, 2), (0, 5), (2, 5)]


def test_iter_faces_vertices_of_tetrahedron() -> None:
    tetra = (3, 4, 6, 9)
    vertices = list(iter_faces(tetra, face_dim=0))
    assert vertices == [(3,), (4,), (6,), (9,)]