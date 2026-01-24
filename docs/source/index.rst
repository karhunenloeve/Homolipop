Homolipop
=========

Homolipop is a research oriented Python package for computational geometry and
topological data analysis. The current scope covers Delaunay triangulations in
:math:`\mathbb{R}^d`, alpha filtrations, coboundary operators with ring
coefficients, and persistent homology over fields.

Core workflow
-------------

1. Start with a point cloud in :math:`\mathbb{R}^d`.
2. Compute a Delaunay triangulation and its simplicial closure.
3. Assign filtration values using the alpha filtration.
4. Construct coboundary or boundary operators with chosen coefficients.
5. Compute persistence pairs over a field.

Modules
-------

- ``homolipop.delaunay``: Delaunay triangulation via paraboloid lifting and convex hull
- ``homolipop.simplices``: simplicial closure, faces, indexing
- ``homolipop.alpha``: alpha filtration values, monotonicity propagation
- ``homolipop.filtration``: deterministic filtration order
- ``homolipop.coboundary``: coboundary operators over general rings
- ``homolipop.persistence``: persistent homology over fields, unit pivot reduction mode

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   api

Notes
-----

The API reference is generated from docstrings via Sphinx autodoc. If source
links are desired, enable ``sphinx.ext.viewcode`` in ``conf.py``.