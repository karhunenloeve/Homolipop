Homolipop
=========

Homolipop is a research oriented Python package for computational geometry,
topological data analysis, and operator-algebraic persistence.

The package provides explicit pipelines from point clouds to invariants derived
from algebraic topology and C*-algebra K-theory, with an emphasis on
mathematical correctness and reproducibility.

Overview
--------

Given a finite set of points in :math:`\mathbb{R}^d`, Homolipop supports:

- construction of Delaunay triangulations and alpha complexes
- deterministic filtrations compatible with geometric scale
- persistent homology over fields and barcode extraction
- operator-algebraic K-theory invariants derived from graph C*-algebras
- visualization of homological and K-theoretic persistence data

Homological pipeline
--------------------

1. Build a Delaunay triangulation in :math:`\mathbb{R}^d`
2. Construct the associated alpha filtration
3. Order simplices compatibly with the filtration
4. Compute persistent homology over a field
5. Extract and visualize barcodes

Operator-algebraic K-theory pipeline
------------------------------------

Graph filtration from a point cloud
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`X = \{x_1,\dots,x_n\} \subset \mathbb{R}^d`. For a scale
:math:`t \ge 0`, define an undirected proximity graph :math:`G_t` with vertex
set :math:`\{1,\dots,n\}` and an edge :math:`\{i,j\}` whenever
:math:`\lVert x_i - x_j \rVert \le t`. The graphs form a filtration
:math:`G_s \subseteq G_t` for :math:`s \le t`.

Cuntz–Krieger algebras and K-theory over :math:`\mathbb{F}_p`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Choose a deterministic orientation of :math:`G_t` and let :math:`A_t` be the
resulting :math:`n \times n` adjacency matrix with entries in :math:`\{0,1\}`.
Let :math:`O_{A_t}` denote the associated Cuntz–Krieger C*-algebra.

For a prime :math:`p`, Homolipop computes the vector spaces
:math:`K_i(O_{A_t}) \otimes \mathbb{F}_p` using the standard kernel–cokernel
presentations induced by the integer matrix :math:`I - A_t^{\mathsf T}`.
After base change to :math:`\mathbb{F}_p`, these become linear algebra problems:

- :math:`K_1(O_{A_t}) \otimes \mathbb{F}_p \cong \ker_{\mathbb{F}_p}(I - A_t^{\mathsf T})`
- :math:`K_0(O_{A_t}) \otimes \mathbb{F}_p \cong \operatorname{coker}_{\mathbb{F}_p}(I - A_t^{\mathsf T})`

Over the field :math:`\mathbb{F}_p`, one has
:math:`\dim_{\mathbb{F}_p}\operatorname{coker}(M) = n - \operatorname{rank}(M)`.

K-theory profiles and persistence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Homolipop exposes the evolution of
:math:`\dim_{\mathbb{F}_p}(K_0(O_{A_t}) \otimes \mathbb{F}_p)` and
:math:`\dim_{\mathbb{F}_p}(K_1(O_{A_t}) \otimes \mathbb{F}_p)` along the scale
parameter :math:`t` as a K-theory profile.

Full barcode style persistence requires functorial star homomorphisms
:math:`O_{A_s} \to O_{A_t}` inducing linear maps on
:math:`K_i \otimes \mathbb{F}_p`. Constructing these maps canonically for the
chosen graph model is an active direction of development.

Quick usage
-----------

The high level pipelines are available as:

- ``persistent_homology_from_points`` for alpha based persistent homology
- ``k_theory_profile_from_points`` for operator-algebraic K-theory profiles

See the ``examples`` directory for complete runnable scripts.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   api

Notes
-----

The API reference is generated automatically from source code docstrings using
Sphinx autodoc. Source code links are available via ``sphinx.ext.viewcode``.