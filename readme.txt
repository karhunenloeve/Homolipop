Homolipop
=========

Homolipop is a research oriented Python package for computational geometry,
topological data analysis, and operator-algebraic persistence.

The package provides a coherent pipeline from point clouds to invariants derived
from algebraic topology and C*-algebra K-theory, with a strong emphasis on
mathematical correctness, explicit constructions, and reproducibility.

-----------------------------------------------------------------------
Scope
-----------------------------------------------------------------------

Homolipop currently supports:

Geometric constructions
- Delaunay triangulations in R^d via paraboloid lifting
- Alpha complexes and alpha filtrations
- Deterministic filtration orders

Algebraic topology
- Sparse coboundary operators over arbitrary rings
- Persistent homology over fields
- Barcode extraction and visualization

Operator algebraic persistence
- Graph filtrations induced by proximity in point clouds
- Cuntz–Krieger graph C*-algebras O_A
- Computation of K_0 ⊗ F_p and K_1 ⊗ F_p via kernel/cokernel presentations
- Persistent K-theory profiles along graph filtrations
- Visualization of K-theoretic invariants along scale parameters

-----------------------------------------------------------------------
Mathematical background
-----------------------------------------------------------------------

Persistent homology is computed over fields, where persistence modules admit
interval decompositions and barcodes.

Persistent K-theory is implemented in an operator-algebraic sense by associating
to each filtration step a graph C*-algebra and computing its K-theory after base
change to a finite field F_p. This yields computable, stable invariants that are
functorial at the level of vector spaces.

At present, Homolipop provides K-theory profiles along filtrations. The
construction of full K-theory barcodes requires explicit functorial *-homomorphisms
between graph C*-algebras and is an active direction of development.

-----------------------------------------------------------------------
Documentation
-----------------------------------------------------------------------

Online documentation:
https://karhunenloeve.github.io/Homolipop/

API reference:
https://karhunenloeve.github.io/Homolipop/api.html

The documentation is generated from source code docstrings using Sphinx and
published automatically via GitHub Pages.

-----------------------------------------------------------------------
Installation
-----------------------------------------------------------------------

Editable development install:
    python -m pip install -e ".[dev]"

Runtime-only install:
    python -m pip install -e .

Optional visualization dependencies:
    python -m pip install ".[viz]"

-----------------------------------------------------------------------
Examples
-----------------------------------------------------------------------

Runnable examples are located in the examples directory.

Persistent homology with alpha filtration:
    python examples/pipeline_alpha_persistence_plot.py

Operator-algebraic K-theory profile for graph filtrations:
    python examples/pipeline_kgraph_profile_plot.py

-----------------------------------------------------------------------
Project layout
-----------------------------------------------------------------------

src/homolipop/
  alpha.py
  barcodes.py
  coboundary.py
  delaunay.py
  filtration.py
  graph_filtration.py
  kgraph.py
  kplotting.py
  persistence.py
  plotting.py
  simplices.py
  pipeline.py

examples/
tests/
docs/

-----------------------------------------------------------------------
Testing
-----------------------------------------------------------------------

Run the test suite from the repository root:
    pytest

-----------------------------------------------------------------------
Roadmap
-----------------------------------------------------------------------

Planned extensions include:
- Functorial persistent K-theory barcodes for graph C*-algebras
- Multi-prime torsion-sensitive K-theory analysis
- Integration with noncommutative metric geometry constructions
- Spectral sequence based comparisons between homology and K-theory persistence

-----------------------------------------------------------------------
License
-----------------------------------------------------------------------

MIT License

-----------------------------------------------------------------------
Author
-----------------------------------------------------------------------

Luciano Melodia