Homolipop
=========

Homolipop is a research oriented Python package for computational geometry and topological data analysis.
It provides explicit, reproducible pipelines from finite metric data, such as point clouds and graphs, to field valued persistence invariants, together with lightweight visualization utilities.

The implementation favors explicit conventions, deterministic outputs, and mathematically checkable constructions.

-----------------------------------------------------------------------
Features.
-----------------------------------------------------------------------

Geometry.
- Delaunay triangulations in R^d via paraboloid lifting and lower convex hull facets.
- Simplicial closure from maximal simplices, and truncation to a fixed skeleton.
- Alpha squared values, and alpha filtration compatible simplex orders.

Algebraic topology and persistence.
- Sparse coboundary operators over rings.
- Persistent homology over fields via standard reduction.
- Barcode extraction, and plotting helpers.

Graph persistence.
- Graph filtrations induced by proximity in point clouds.
- Deterministic orientations of undirected graphs to directed filtrations.
- Functorial graph persistence over F_p with H_0 and H_1 barcodes.

Toeplitz motivated quotients, and K0 like persistence.
- A computable field linear invariant motivated by Toeplitz quotient semantics.
- Persistence in the quotient direction implemented by reversing the filtration, and applying field persistence reduction.

Coarse geometry prototypes.
- A computable proxy for coarse stabilization on finite metric inputs using Vietoris–Rips complexes at increasing scale.
- An example comparing this stabilization proxy with ordinary persistence.

Visualization.
- simplicialviz provides Matplotlib based rendering of point clouds and simplicial complexes up to dimension 3.

-----------------------------------------------------------------------
Mathematical conventions.
-----------------------------------------------------------------------

Persistent homology is computed over a field F, so that the persistence modules are pointwise finite dimensional and admit interval decompositions.
Homolipop returns barcodes in the usual birth death convention.

Graph persistence over F_p.
- For a filtered directed graph E_t on a fixed vertex set V, define C_1(t) = F_p^{E_t}, C_0(t) = F_p^{V}, and ∂(u→v) = e_v − e_u.
- Inclusions E_s ⊆ E_t induce chain maps, and therefore induced maps on H_0 and H_1.
- Barcodes are computed by reduction over F_p.

Toeplitz quotient direction, and K0 like persistence.
- Let E_max be the union of edges across the filtration.
- Removing edges defines ideals I_t and quotients A_t = T(E_max)/I_t, with surjections A_t → A_s for s ≤ t.
- Homolipop implements the field linear surrogate K0_like(t) = F_p^V / ⟨ e_v − e_u : (u→v) ∈ E_t ⟩, which is canonically isomorphic to H_0(E_t; F_p).
- Persistence in the quotient direction is computed by reversing the filtration.

Coarse geometry proxy.
- For a metric space X, a standard computable proxy for coarse stabilization is the direct limit of homology of Rips complexes as the scale r → ∞.
- For finite bounded point clouds, this stabilizes to the homology of a point.
- Homolipop currently provides a computable finite input proxy, and a comparison example, not a full axiomatic bornological coarse homology implementation.

Clarifications.
- No persistent K theory of C star algebras is computed.
- The coarse component is a finite input stabilization proxy.

-----------------------------------------------------------------------
Installation.
-----------------------------------------------------------------------

Editable development install.
    python -m pip install -e ".[dev]"

Runtime install.
    python -m pip install -e .

Optional visualization extras.
    python -m pip install ".[viz]"

-----------------------------------------------------------------------
Examples.
-----------------------------------------------------------------------

All examples are runnable scripts in examples/.

Alpha filtration persistence.
    python examples/pipeline_alpha_persistence_plot.py

Graph persistence and profiling.
    python examples/pipeline_kgraph_profile_plot.py

K0 like persistence.
    python examples/pipeline_points_to_k0_like_plot.py
    python examples/pipeline_toeplitz_k_theory_barcodes_plot.py

3D visualization.
    python examples/simplicialviz_points_and_complex_3d.py

Coarse stabilization proxy vs persistence.
    python examples/coarse_vs_persistent_homology.py

-----------------------------------------------------------------------
Project layout.
-----------------------------------------------------------------------

examples/.
  Runnable scripts.

src/homolipop/.
  Core library code.
  distances/ contains distance related algorithms.

src/simplicialviz/.
  Visualization utilities.

tests/.
  Pytest based test suite.

-----------------------------------------------------------------------
Testing.
-----------------------------------------------------------------------

From the repository root.
    pytest

-----------------------------------------------------------------------
License.
-----------------------------------------------------------------------

MIT License.

-----------------------------------------------------------------------
Author.
-----------------------------------------------------------------------

Luciano Melodia.