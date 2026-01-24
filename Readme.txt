Homolipop
===========

Homolipop is a research oriented Python package for computational geometry and
topological data analysis. The current implementation provides

- Delaunay triangulation in R^d via paraboloid lifting and convex hull
- Explicit simplicial complex construction via simplicial closure
- Alpha filtration values based on circumsphere radii with monotonic face propagation
- Coboundary operators with coefficients in general rings, including Z and F_p

The intended downstream objective is persistent cohomology.

-----------------------------------------------------------------------
Mathematical conventions and data representations
-----------------------------------------------------------------------

Point clouds
- A point cloud is a NumPy array of shape (n, d) with entries in R.

Simplices
- A simplex is represented as a strictly increasing tuple of vertex indices.
  The dimension is dim(σ) = |σ| - 1.
  Examples
    (i,)           0 simplex
    (i, j)         1 simplex
    (i, j, k)      2 simplex
    (i, j, k, l)   3 simplex

Simplicial complexes
- Given a set of maximal simplices, the package forms the simplicial closure,
  i.e. the set of all faces, optionally truncated at a prescribed dimension.

Orientations
- Each simplex σ = (v0, ..., vk) is oriented by the increasing vertex order.
- Incidence coefficients are determined by the alternating sign convention
  in the boundary operator.

Filtration values
- Alpha filtration values are stored as squared radii.
- Monotonicity is enforced by propagating coface values down to faces.

-----------------------------------------------------------------------
Modules
-----------------------------------------------------------------------

delaunay.py
- Delaunay triangulation in R^d via lifting to (x, ||x||^2) in R^(d+1) and
  extracting lower hull facets.

simplices.py
- Canonical simplex representation and operations.
- Face enumeration.
- Simplicial closure and explicit complex indexing.

alpha.py
- Circumsphere radius computation in the affine hull of a simplex.
- Alpha value assignment and monotonic propagation to faces.

filtration.py
- Construction of a filtration compatible total order on simplices:
    (alpha_sq(σ), dim(σ), lex(σ)).

coboundary.py
- Sparse coboundary operator δ = ∂^T over a general coefficient ring.
- Convenience constructors for Z and F_p.

-----------------------------------------------------------------------
Algorithmic guarantees
-----------------------------------------------------------------------

- Explicit output lower bounds apply throughout: any method enumerating all
  simplices or all incidences must take Omega(output_size) time.
- Delaunay computation inherits output sensitivity from the convex hull routine.
- Alpha propagation performs one update per simplex-face incidence.
- Filtration ordering is comparison optimal at Theta(|K| log |K|).

-----------------------------------------------------------------------
Installation
-----------------------------------------------------------------------

Editable install for development

    python -m pip install -e ".[dev]"

-----------------------------------------------------------------------
Usage sketch
-----------------------------------------------------------------------

Delaunay triangulation in R^d

    import numpy as np
    from homolipop.delaunay import delaunay_d_dim

    points = np.random.default_rng(0).random((50, 2))
    dt = delaunay_d_dim(points)
    top_simplices = dt.delaunay_simplices

Alpha values on the induced complex up to dimension 2

    from homolipop.alpha import alpha_values_squared

    alpha = alpha_values_squared(points, top_simplices, max_dim=2)

Coboundary over Z or F_p

    from homolipop.coboundary import build_coboundary_Z, build_coboundary_Fp

    cobZ = build_coboundary_Z(filtration_simplices)
    cob5 = build_coboundary_Fp(filtration_simplices, p=5)

-----------------------------------------------------------------------
Testing
-----------------------------------------------------------------------

Run the full test suite

    pytest

-----------------------------------------------------------------------
License
-----------------------------------------------------------------------

MIT License

-----------------------------------------------------------------------
Author
-----------------------------------------------------------------------

Luciano Melodia