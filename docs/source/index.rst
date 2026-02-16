Homolipop
=========

Homolipop is a research oriented Python package for computational geometry,
topological data analysis, and graph based persistence models motivated by
operator algebra.

The package provides explicit, deterministic pipelines from finite point clouds
to persistence invariants over finite fields, with an emphasis on mathematical
correctness, functorial constructions, and reproducibility.

Overview
--------

Given a finite set of points in :math:`\mathbb{R}^d`, Homolipop supports

- Delaunay triangulations in :math:`\mathbb{R}^d` via paraboloid lifting and lower convex hull facets
- simplicial complex construction from maximal simplices by downward closure
- squared alpha values :math:`\alpha^2` and filtration compatible simplex orders
- persistent homology over fields and barcode extraction
- barcode plotting
- fixed vertex graph filtrations from point clouds and field valued graph persistence
- graph C star algebra motivated quotient direction models producing finite field persistence invariants

A crucial point
---------------

Homolipop does not compute persistent K theory.

In the operator algebra motivated part, Homolipop computes functorial persistence
modules over finite fields derived from filtered graphs and quotient direction
semantics. These invariants are well defined and computable. They are intended as
finite field linear surrogates motivated by graph C star algebra presentations.

Homological pipeline
--------------------

Given :math:`X = \{x_0,\dots,x_{n-1}\} \subset \mathbb{R}^d`

1. compute a Delaunay triangulation of :math:`X`
2. build the simplicial complex given by the downward closure of the maximal Delaunay simplices up to a chosen dimension
3. compute squared alpha values :math:`\alpha^2(\sigma)` for all simplices in this complex
4. produce a filtration compatible total order on simplices by sorting by the key
   :math:`\alpha^2(\sigma)`, then simplex dimension, then lexicographic vertex order
5. build the boundary operator over a field and run standard persistence reduction
6. extract barcodes and plot them

The tie break by dimension ensures the following.
If :math:`\tau \subseteq \sigma` and :math:`\alpha^2(\tau)=\alpha^2(\sigma)`,
then :math:`\dim \tau < \dim \sigma`, hence :math:`\tau` appears earlier in the order.
Therefore the order is admissible for persistence.

Graph filtration from a point cloud
-----------------------------------

Fixed vertex edge filtration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`X = \{x_0,\dots,x_{n-1}\} \subset \mathbb{R}^d` and let
:math:`t_0 \le \dots \le t_{T-1}` be thresholds. For each step
:math:`s \in \{0,\dots,T-1\}`, define an undirected graph :math:`G_s` on vertex set
:math:`V = \{0,\dots,n-1\}` by

.. math::

   \{i,j\} \in E(G_s)
   \iff
   d(x_i,x_j) \le t_s,

where :math:`d` is the Euclidean distance or the squared Euclidean distance,
depending on the configuration.

Then :math:`s \le t` implies :math:`E(G_s) \subseteq E(G_t)`, hence
:math:`G_s \subseteq G_t` is a monotone filtration.

For reproducibility, Homolipop optionally applies a deterministic relabeling of
vertices by sorting points by squared distance to the barycenter, with the original
index as a final tie break. This changes only labels. It preserves the isomorphism
type of the filtered graphs.

Efficient representation
^^^^^^^^^^^^^^^^^^^^^^^^

All pairwise distances are computed once. All edges are sorted once by distance.
Each filtration step is represented by a prefix length of the sorted edge list.
Adjacency matrices are materialized only on demand.

This is optimal under the explicit output model.

- Computing all pairwise distances is :math:`\Omega(n^2)` in the worst case.
- Supporting many thresholds with incremental updates requires sorting edge distances
  in the comparison model.
- Materializing a dense adjacency matrix is :math:`\Theta(n^2)` per step.

Directed orientation
^^^^^^^^^^^^^^^^^^^^

To obtain a directed graph filtration, an undirected adjacency matrix can be
oriented deterministically. One option is to direct each edge from lower to higher
vertex index. Another option is the reverse orientation. Optionally both directions
can be included.

Fully functorial graph persistence over :math:`\mathbb{F}_p`
------------------------------------------------------------

Fix a prime :math:`p`. For each step :math:`s` with directed edge set :math:`E_s`
on the fixed vertex set :math:`V`, define a two term chain complex over
:math:`\mathbb{F}_p`

.. math::

   C_1(s) = \mathbb{F}_p^{E_s},
   \qquad
   C_0(s) = \mathbb{F}_p^{V},
   \qquad
   \partial(e_{u \to v}) = e_v - e_u.

For :math:`r \le s`, the inclusion :math:`E_r \subseteq E_s` induces canonical chain maps.

- :math:`C_0(r) \to C_0(s)` is the identity on :math:`\mathbb{F}_p^{V}`.
- :math:`C_1(r) \to C_1(s)` is the basis inclusion on edges.

Hence there are induced linear maps on homology :math:`H_k` for :math:`k \in \{0,1\}`.
Over a field, these persistence modules admit interval decompositions. Homolipop
computes barcodes for :math:`H_0` and :math:`H_1` via standard column reduction.

Interpretation.

- :math:`H_0` persistence is persistent connected components.
- :math:`H_1` persistence is persistent cycle space.

Toeplitz quotient direction semantics and K0 like persistence
-------------------------------------------------------------

Quotient system model
^^^^^^^^^^^^^^^^^^^^^

Let :math:`E_{\max}` be the union of all directed edges occurring in the filtration.
Let :math:`\mathcal{T}(E_{\max})` denote the Toeplitz graph C star algebra.
For each step :math:`s`, define the ideal :math:`I_s` generated by edge generators
corresponding to edges not contained in :math:`E_s`, and set

.. math::

   \mathcal{A}_s = \mathcal{T}(E_{\max}) / I_s.

For :math:`r \le s` one has :math:`I_r \supseteq I_s`, hence there are canonical
surjections

.. math::

   \pi_{s \to r} : \mathcal{A}_s \twoheadrightarrow \mathcal{A}_r.

This points in the reverse direction of the edge inclusion filtration and is
naturally a co persistence system. Homolipop models this direction by reversing
the filtration order when computing field barcodes.

K0 like invariant over :math:`\mathbb{F}_p`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Homolipop exposes a computable K0 like field valued invariant defined directly
from the filtered directed graph.

For each step :math:`s` define

.. math::

   K0_{\mathrm{like}}(s)
   =
   \mathbb{F}_p^{V} \big/ \langle e_v - e_u : (u \to v) \in E_s \rangle.

This vector space is canonically isomorphic to
:math:`\mathbb{F}_p^{V} / \operatorname{im}(\partial)` for the chain complex above,
hence to the zeroth homology group :math:`H_0` of that complex. In particular, it is
functorial for edge inclusions by the identity map on vertices, hence admits barcodes
over :math:`\mathbb{F}_p`.

To model the Toeplitz quotient direction, Homolipop computes persistence in the
reverse filtration direction, corresponding to maps from later to earlier steps.

Coboundary with ring coefficients
---------------------------------

Homolipop provides a sparse coboundary operator on simplicial complexes with
coefficients in an arbitrary ring :math:`R` specified by operations ``one``, ``add``,
``neg``, ``is_zero``.

For an ordered simplex :math:`(v_0,\dots,v_{k+1})`, the :math:`i` th face is obtained
by deleting :math:`v_i`. The coboundary on cochains is defined by the alternating sum

.. math::

   (\delta_k \varphi)(v_0,\dots,v_{k+1})
   =
   \sum_{i=0}^{k+1} (-1)^i \varphi(v_0,\dots,\widehat{v_i},\dots,v_{k+1}).

No multiplication or division is needed to build the sparse coboundary itself.

Plotting
--------

Barcode plotting supports

- empty inputs
- infinite intervals
- degenerate spans
- optional rendering of zero length intervals

This avoids false empty plots in filtrations where births and deaths can occur at
the same filtration value.

Quick usage
-----------

Homological alpha pipeline.

- compute Delaunay triangulation
- compute alpha filtration order
- compute persistent homology over :math:`\mathbb{F}_p`
- extract barcodes and plot

Graph based pipelines.

- build a fixed vertex edge filtration from points
- orient to a directed filtration
- compute functorial graph persistence over :math:`\mathbb{F}_p`
- compute K0 like persistence in Toeplitz quotient direction over :math:`\mathbb{F}_p`

See the examples directory for complete runnable scripts.

Contents
========

.. toctree::
   :maxdepth: 3
   :titlesonly:

   api
   examples
   tests

Notes
-----

The API reference is generated automatically from source code docstrings using
Sphinx autodoc. Source code links are available via ``sphinx.ext.viewcode``.