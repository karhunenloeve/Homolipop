Homolipop
=========

Homolipop is a research oriented Python package for computational geometry,
topological data analysis, and graph based persistence models inspired by
operator algebra.

The package provides explicit, deterministic pipelines from point clouds to
field valued persistence invariants, with an emphasis on mathematical
correctness, functorial constructions, and reproducibility.

Overview
--------

Given a finite set of points in :math:`\mathbb{R}^d`, Homolipop supports

- Delaunay triangulations in :math:`\mathbb{R}^d` via paraboloid lifting and lower convex hull facets
- simplicial complex construction from maximal simplices and simplicial closure
- alpha squared values and filtration compatible simplex orders
- persistent homology over fields and barcode extraction
- robust barcode plotting
- fixed vertex graph filtrations from point clouds and field valued graph persistence
- Toeplitz graph algebra motivated quotient direction persistence models over :math:`\mathbb{F}_p`

A crucial point
---------------

Homolipop does not compute persistent K theory.

What is computed in the operator algebra inspired part are functorial persistence
modules over finite fields derived from filtered graphs and quotient direction
semantics. These are mathematically well defined and computable, and are intended
as field linear surrogates motivated by graph C star algebra presentations.

Homological pipeline
--------------------

Given :math:`X = \{x_0,\dots,x_{n-1}\} \subset \mathbb{R}^d`

1. compute the Delaunay triangulation of :math:`X`
2. compute alpha squared values for simplices up to a chosen dimension
3. produce a filtration compatible total order on simplices by the key
   alpha squared, then simplex dimension, then lexicographic vertex order
4. build the boundary operator over a field and run standard persistence reduction
5. extract barcodes and plot them

The tie break by dimension ensures that if a face and a coface share the same
alpha value then the face appears earlier, hence the ordering is compatible with
the filtration.

Graph filtration from a point cloud
-----------------------------------

Fixed vertex edge filtration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`X = \{x_0,\dots,x_{n-1}\} \subset \mathbb{R}^d` and let
:math:`t_0 \le \dots \le t_{T-1}` be thresholds. Define an undirected graph
:math:`G_t` on vertex set :math:`V = \{0,\dots,n-1\}` by the proximity rule

:math:`\{i,j\} \in E(G_t)` if and only if :math:`d(x_i,x_j) \le t`.

This yields a monotone filtration :math:`G_{t_s} \subseteq G_{t_t}` for
:math:`s \le t`.

For reproducibility, Homolipop optionally applies a deterministic relabeling of
vertices by sorting points by squared distance to the barycenter, with index as
a final tie break. This changes only the labels, not the abstract filtered graph
up to isomorphism.

Efficient representation
^^^^^^^^^^^^^^^^^^^^^^^^

All pairwise distances are computed once, all edges are sorted once by distance,
and each filtration step is represented by a prefix length of the sorted edge
list. Adjacency matrices are materialized only on demand.

This is optimal under the explicit output model
- computing all pairwise distances is :math:`\Omega(n^2)` in the worst case
- supporting many thresholds with incremental updates requires sorting edge distances in the comparison model
- materializing a dense adjacency matrix is :math:`\Theta(n^2)` per step

Directed orientation
^^^^^^^^^^^^^^^^^^^^

To obtain a directed graph filtration, an undirected adjacency matrix can be
oriented deterministically, for example by directing each edge from lower to
higher vertex index, or the reverse. Optionally both directions can be included.

Fully functorial graph persistence over :math:`\mathbb{F}_p`
------------------------------------------------------------

Fix a prime :math:`p`. For each step t with directed edge set :math:`E_t` on the
fixed vertex set :math:`V`, define the 2 term chain complex over :math:`\mathbb{F}_p`

:math:`C_1(t) = \mathbb{F}_p^{E_t}`
:math:`C_0(t) = \mathbb{F}_p^{V}`
:math:`\partial(u \to v) = e_v - e_u`

For :math:`s \le t`, inclusion :math:`E_s \subseteq E_t` induces canonical chain
maps
- :math:`C_0(s) \to C_0(t)` is the identity on :math:`\mathbb{F}_p^{V}`
- :math:`C_1(s) \to C_1(t)` is the basis inclusion on edges

Hence induced linear maps on homology :math:`H_k` for :math:`k \in \{0,1\}`.
Over a field, these persistence modules admit interval decompositions, and
Homolipop computes barcodes for :math:`H_0` and :math:`H_1` via standard column
reduction.

Interpretation
- :math:`H_0` persistence is persistent connected components
- :math:`H_1` persistence is persistent cycle space
both with :math:`\mathbb{F}_p` coefficients

Toeplitz quotient direction semantics and K0 like persistence
-------------------------------------------------------------

Quotient system model
^^^^^^^^^^^^^^^^^^^^^

Let :math:`E_{\max}` be the union of all edges occurring in the directed
filtration. Let :math:`\mathcal{T}(E_{\max})` denote the Toeplitz graph C star
algebra. For each t, define the ideal :math:`I_t` generated by edge generators
corresponding to edges not contained in :math:`E_t` and set

:math:`\mathcal{A}_t = \mathcal{T}(E_{\max}) / I_t`.

For :math:`s \le t` one has :math:`I_s \supseteq I_t`, hence there are canonical
surjections

:math:`\pi_{t \to s} : \mathcal{A}_t \twoheadrightarrow \mathcal{A}_s`.

This points in the reverse direction of the edge inclusion filtration and is
naturally a co persistence system. Homolipop models this direction by reversing
the filtration order when computing field barcodes.

K0 like invariant over :math:`\mathbb{F}_p`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Homolipop exposes a computable K0 like field valued invariant defined directly
from the filtered directed graph.

For each t define

:math:`K0\_like(t) = \mathbb{F}_p^{V} / \langle e_v - e_u : (u \to v) \in E_t \rangle`.

This vector space is canonically isomorphic to :math:`H_0(E_t;\mathbb{F}_p)` of
the graph chain complex above. In particular, it is functorial for edge
inclusions by the identity map on vertices, hence admits barcodes over
:math:`\mathbb{F}_p`.

To model the Toeplitz quotient direction, Homolipop computes persistence in the
reverse filtration direction, corresponding to maps from later to earlier steps.

Coboundary with ring coefficients
---------------------------------

Homolipop also provides a sparse coboundary operator on simplicial complexes
with coefficients in an arbitrary ring R specified by operations one, add, neg,
is zero. Orientation is fixed by increasing vertex order, and incidence signs
are :math:`(-1)^i` for deleting the i th vertex.

No multiplication or division is needed to build the coboundary itself.

Plotting
--------

Barcode plotting supports
- empty inputs
- infinite intervals
- degenerate spans
- optional rendering of zero length intervals

This avoids false empty plots in filtrations where births and deaths can occur
at the same filtration value.

Quick usage
-----------

Homological alpha pipeline
- compute Delaunay triangulation
- compute alpha filtration order
- compute persistent homology over :math:`\mathbb{F}_p`
- extract barcodes and plot

Graph based pipelines
- build a fixed vertex edge filtration from points
- orient to a directed filtration
- compute functorial graph persistence over :math:`\mathbb{F}_p`
- compute K0 like persistence in Toeplitz quotient direction over :math:`\mathbb{F}_p`

See the examples directory for complete runnable scripts.

Contents
--------

.. toctree::
   :maxdepth: 3
   :caption: Documentation
   :titlesonly:

   api

Notes
-----

The API reference is generated automatically from source code docstrings using
Sphinx autodoc. Source code links are available via sphinx.ext.viewcode.