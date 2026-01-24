Homolipop
=========

Homolipop is a research oriented Python package for computational geometry,
topological data analysis, and graph based persistence models inspired by
operator algebra.

The package provides explicit pipelines from point clouds to field valued
persistence invariants, with a strong emphasis on mathematical correctness,
functorial constructions, explicit conventions, and reproducibility.

-----------------------------------------------------------------------
Scope
-----------------------------------------------------------------------

Homolipop currently supports

Geometric constructions
- Delaunay triangulations in R^d via paraboloid lifting and lower convex hull facets
- Simplicial complex construction from maximal simplices and simplicial closure
- Alpha squared values and alpha filtration compatible simplex orders

Algebraic topology
- Sparse coboundary operators over arbitrary rings
- Persistent homology over fields
- Barcode extraction and visualization

Graph based persistence
- Fixed vertex graph filtrations induced by proximity in point clouds
- Deterministic orientations of undirected graphs to directed filtrations
- Fully functorial graph persistence over F_p with H_0 and H_1 barcodes

Toeplitz quotient direction models
- A Toeplitz graph algebra motivated quotient semantics for edge filtrations
- A computable K0 like invariant over F_p with persistence in the quotient direction
- End to end pipelines from points to K0 like barcodes over F_p

-----------------------------------------------------------------------
Mathematical background
-----------------------------------------------------------------------

Persistent homology is computed over fields, where pointwise finite dimensional
persistence modules admit interval decompositions and barcodes.

Graph persistence over F_p
- Fix a prime p and a filtered directed graph E_t on a fixed vertex set V.
- Define C_1(t) = F_p^{E_t}, C_0(t) = F_p^{V} and ∂(u→v) = e_v − e_u.
- For s ≤ t, inclusions E_s ⊆ E_t induce canonical chain maps, hence induced maps
  on H_0 and H_1.
- Over a field, these persistence modules admit barcode decompositions and
  Homolipop computes the H_0 and H_1 barcodes by standard column reduction.

Toeplitz quotient direction and K0 like persistence
- Let E_max be the union of edges across the filtration and let T(E_max) denote
  the Toeplitz graph C star algebra.
- Removing edges corresponds to ideals I_t and quotient algebras A_t = T(E_max)/I_t,
  giving canonical surjections π_{t→s}: A_t → A_s for s ≤ t.
- Homolipop models this quotient direction at the level of a computable field
  valued invariant
  K0_like(t) = F_p^V / < e_v − e_u : (u→v) ∈ E_t >,
  canonically isomorphic to H_0(E_t; F_p).
- Persistence barcodes in the quotient direction are computed by reversing the
  filtration order and applying field persistence reduction.

Important clarification
- Homolipop does not compute persistent K theory of C star algebras.
- No persistence module {K_i(A_t), (f_{s,t})_*} arising from actual * homomorphisms
  is computed at present.
- The operator algebra content currently implemented is a mathematically correct,
  field linear persistence surrogate motivated by Toeplitz quotient semantics.

-----------------------------------------------------------------------
Documentation
-----------------------------------------------------------------------

Online documentation
https://karhunenloeve.github.io/Homolipop/

API reference
https://karhunenloeve.github.io/Homolipop/api.html

The documentation is generated from source code docstrings using Sphinx and
published automatically via GitHub Pages.

-----------------------------------------------------------------------
Installation
-----------------------------------------------------------------------

Editable development install
    python -m pip install -e ".[dev]"

Runtime only install
    python -m pip install -e .

-----------------------------------------------------------------------
Examples
-----------------------------------------------------------------------

Runnable examples are located in the examples directory.

Persistent homology with alpha filtration
    python examples/pipeline_alpha_persistence_plot.py

Graph persistence over F_p
    python examples/pipeline_points_to_graph_persistence_plot.py

K0 like persistence in Toeplitz quotient direction over F_p
    python examples/pipeline_points_to_k0_like_plot.py

-----------------------------------------------------------------------
Project layout
-----------------------------------------------------------------------

src/homolipop/
  alpha.py
  barcodes.py
  coboundary.py
  delaunay.py
  filtration.py
  graph_persistence_fp.py
  edge_filtration.py
  k0_like.py
  pipeline_k0_like.py
  persistence.py
  plotting.py
  simplices.py
  cstar_functor.py

examples/
tests/
docs/

-----------------------------------------------------------------------
Testing
-----------------------------------------------------------------------

Run the test suite from the repository root
    pytest

-----------------------------------------------------------------------
Roadmap
-----------------------------------------------------------------------

Planned extensions include
- A genuine persistence module {K_i(A_t), (f_{s,t})_*} coming from explicit * homomorphisms
- Computable induced maps on K theory for nontrivial graph C star algebra models
- Comparisons between graph homology persistence and operator algebra motivated invariants
- Torsion sensitive invariants beyond field coefficients

-----------------------------------------------------------------------
License
-----------------------------------------------------------------------

MIT License

-----------------------------------------------------------------------
Author
-----------------------------------------------------------------------

Luciano Melodia