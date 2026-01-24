Homolipop
=========

Homolipop is a research oriented Python package for computational geometry and
topological data analysis. The current scope covers Delaunay triangulations in
R^d, alpha filtrations, coboundary operators with ring coefficients, and
persistent homology over fields.

-----------------------------------------------------------------------
Documentation
-----------------------------------------------------------------------

Online documentation
- https://karhunenloeve.github.io/Homolipop/

API reference
- https://karhunenloeve.github.io/Homolipop/api.html

The documentation is generated from source code docstrings using Sphinx and is
published via GitHub Pages.

-----------------------------------------------------------------------
Installation
-----------------------------------------------------------------------

Development install
    python -m pip install -e ".[dev]"

Runtime-only install
    python -m pip install -e .

-----------------------------------------------------------------------
Examples
-----------------------------------------------------------------------

Runnable example scripts are located in the examples directory.

Run an example from the repository root
    python examples/persistence_triangle_F2.py

If you see ModuleNotFoundError, install the package in editable mode first.

-----------------------------------------------------------------------
Testing
-----------------------------------------------------------------------

Run the full test suite from the repository root
    pytest

-----------------------------------------------------------------------
Project layout
-----------------------------------------------------------------------

src/homolipop/
  __init__.py
  delaunay.py
  simplices.py
  alpha.py
  filtration.py
  coboundary.py
  persistence.py

examples/
tests/
docs/

-----------------------------------------------------------------------
Publishing documentation
-----------------------------------------------------------------------

The documentation is built and deployed automatically on every push to main.

Enable GitHub Pages
- Repository Settings -> Pages -> Source: GitHub Actions

Workflow
- .github/workflows/docs.yml builds docs and deploys them to GitHub Pages.

-----------------------------------------------------------------------
Mathematical scope and limitations
-----------------------------------------------------------------------

- Over fields F_p, persistent homology admits interval decompositions and is
  fully supported by the field reduction algorithm.
- Over general rings, interval decompositions need not exist. Homolipop provides
  coboundary operators over arbitrary rings and a unit pivot reduction mode.
  Full integer persistence with torsion requires Smith normal form style
  algorithms and is not implemented in the current scope.

-----------------------------------------------------------------------
License
-----------------------------------------------------------------------

MIT License

-----------------------------------------------------------------------
Author
-----------------------------------------------------------------------

Luciano Melodia