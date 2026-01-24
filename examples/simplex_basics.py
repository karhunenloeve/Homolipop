from homolipop.simplices import canonical_simplex, iter_faces, simplex_dim


def main() -> None:
    raw = (5, 2, 7, 2)
    simplex = canonical_simplex(raw)

    print("Raw vertices:", raw)
    print("Canonical simplex:", simplex)
    print("Dimension:", simplex_dim(simplex))

    print("All 2-faces (triangles) of the 3-simplex:")
    for face in iter_faces(simplex, face_dim=2):
        print(face)

    print("All 1-faces (edges) of the 3-simplex:")
    for face in iter_faces(simplex, face_dim=1):
        print(face)


if __name__ == "__main__":
    main()