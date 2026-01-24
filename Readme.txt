RobbyBubble
===========

RobbyBubble is a computational geometry and topology library focused on
constructing alpha filtrations and computing persistent cohomology.

The design goal is correctness, mathematical clarity, and asymptotically
optimal algorithms with clean separation of concerns.

-----------------------------------------------------------------------
Core concepts
-----------------------------------------------------------------------

Points
- Input data is a finite set of points in R^d, represented as a NumPy array
  of shape (n_points, ambient_dimension).

Simplex
- A simplex is represented as a strictly increasing tuple of vertex indices.
  Examples:
    (i,)           vertex
    (i, j)         edge
    (i, j, k)      triangle
    (i, j, k, l)   tetrahedron
- The dimension of a simplex is len(simplex) - 1.

Simplicial complex
- A simplicial complex is represented explicitly by all its simplices.
- Given maximal simplices, the simplicial closure contains all faces.

Filtration
- A filtration assigns a nondecreasing real value to simplices such that
  every face appears no later than its cofaces.

-----------------------------------------------------------------------
Modules
-----------------------------------------------------------------------

delaunay.py
- Computes the Delaunay triangulation in arbitrary dimension using the
  paraboloid lifting and convex hull method.
- Output is a list of top-dimensional simplices.

simplices.py
- Canonical simplex representation.
- Face enumeration.
- Construction of the simplicial closure.
- Indexing and grouping by dimension.

alpha.py
- Computes squared alpha values for all simplices.
- Uses circumsphere radii for top-dimensional simplices.
- Enforces filtration monotonicity by propagating values to faces.

The alpha filtration produced here is suitable as input for persistent
(co)homology algorithms.

-----------------------------------------------------------------------
Mathematical guarantees
-----------------------------------------------------------------------

- All algorithms are explicit-output algorithms and therefore optimal up
  to constant factors: runtime is Omega(size of the output).
- Delaunay construction is output-sensitive via convex hull computation.
- Alpha value propagation touches each incidence relation exactly once.
- All simplices are canonical, immutable, and hashable.

-----------------------------------------------------------------------
Quick example
-----------------------------------------------------------------------

Compute a 2D alpha filtration:

    import numpy as np
    from robbybubble.delaunay import delaunay_d_dim
    from robbybubble.alpha import alpha_values_squared

    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    delaunay = delaunay_d_dim(points)
    alpha = alpha_values_squared(points, delaunay.delaunay_simplices, max_dim=2)

    for simplex, value in alpha.alpha_sq.items():
        print(simplex, value)

-----------------------------------------------------------------------
Testing
-----------------------------------------------------------------------

Run all tests from the repository root:

    pytest

Tests cover:
- simplex canonicalization and face enumeration
- simplicial closure correctness
- exact alpha values for simple configurations
- monotonicity of the alpha filtration

-----------------------------------------------------------------------
Intended next steps
-----------------------------------------------------------------------

- Persistent cohomology over F2
- Clearing and compression optimizations
- Barcode and diagram extraction
- Optional sparse and approximate filtrations

-----------------------------------------------------------------------
License
-----------------------------------------------------------------------

MIT License

-----------------------------------------------------------------------
Author
-----------------------------------------------------------------------

Luciano Melodia