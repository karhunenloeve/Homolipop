from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from scipy.spatial import ConvexHull

from .simplices import Simplex


@dataclass(frozen=True)
class DelaunayResult:
    r"""
    Container for the geometric output of a Delaunay construction via convex lifting.

    Mathematical meaning
    ====================

    Let :math:`P = \{p_1,\dots,p_n\} \subset \mathbb{R}^d` be a finite point set.
    Consider the standard paraboloid lifting map

    .. math::

        L : \mathbb{R}^d \to \mathbb{R}^{d+1},
        \quad
        L(x) = (x, \|x\|^2).

    Let :math:`\operatorname{conv}(L(P)) \subset \mathbb{R}^{d+1}` be the convex hull
    of the lifted points. The *lower hull* is the set of facets whose outward normal
    has strictly negative last coordinate. Projecting these facets back to
    :math:`\mathbb{R}^d` yields the (regular) Delaunay triangulation of :math:`P`
    in general position. Each lower facet corresponds to a Delaunay simplex.

    This dataclass stores

    - a list of the resulting Delaunay simplices, expressed as instances of
      :class:`~.simplices.Simplex`
    - the full convex hull in :math:`\mathbb{R}^{d+1}` of the lifted point set,
      represented by :class:`scipy.spatial.ConvexHull`

    Invariants
    ==========
    - ``lifted_hull`` is a convex hull of lifted points in dimension :math:`d+1`
    - each simplex in ``delaunay_simplices`` is typically a :math:`d`-simplex
      from the lower hull projection, but the exact dimension depends on how
      ``Simplex`` encodes simplices and on degeneracies of the input.

    Notes on degeneracies
    =====================
    If the point set is not in general position, the Delaunay structure may not be
    a triangulation but a cell complex with co-spherical sets. In such cases, the
    projection of lower hull faces can yield non-simplicial cells; an implementation
    may triangulate them, choose a consistent refinement, or keep only a subset.
    This container does not enforce a particular convention; it only stores the
    simplices that the upstream algorithm decided to output.
    """

    delaunay_simplices: List[Simplex]
    lifted_hull: ConvexHull


@dataclass(frozen=True)
class AlphaFiltration:
    r"""
    Alpha filtration values for simplices, stored as squared radii.

    Mathematical meaning
    ====================

    Fix a finite point set :math:`P \subset \mathbb{R}^d`. For each scale
    :math:`\alpha \ge 0`, the *alpha complex* :math:`\mathcal{A}_\alpha` is a
    subcomplex of the Delaunay complex consisting of those simplices whose
    *empty circumsphere* has radius at most :math:`\alpha` when restricted by the
    Voronoi cells, equivalently those simplices that appear in the intersection
    of the union of closed balls :math:`\bigcup_{p \in P} \overline{B}(p,\alpha)`
    with the Voronoi diagram.

    A standard equivalent characterization is:

    A simplex :math:`\sigma` with vertices in :math:`P` belongs to
    :math:`\mathcal{A}_\alpha` if and only if there exists a point
    :math:`x \in \mathbb{R}^d` such that

    - :math:`x` is equidistant to all vertices of :math:`\sigma`
    - this common distance is at most :math:`\alpha`
    - and :math:`x` lies in the intersection of the Voronoi cells of the vertices

    The function

    .. math::

        f(\sigma) = \inf\{\alpha^2 : \sigma \in \mathcal{A}_\alpha\}

    is the *filtration value* of the simplex, stored here as a squared radius
    :math:`\alpha^2` for numerical stability and to avoid unnecessary square roots.

    Data model
    ==========
    ``alpha_sq`` maps each simplex :math:`\sigma` to the value :math:`f(\sigma)`.

    Required properties for correctness
    ===================================
    For a valid filtration, the map should be monotone w.r.t. inclusion:

    .. math::

        \tau \subseteq \sigma \implies f(\tau) \le f(\sigma).

    This container does not enforce monotonicity, but downstream persistent
    homology algorithms usually assume it.

    Units and conventions
    =====================
    - values are squared distances in the ambient Euclidean metric
    - ``0.0`` typically corresponds to vertices, since each vertex appears at
      scale :math:`\alpha = 0` in the alpha complex construction.

    Notes
    =====
    In implementations derived from Delaunay circumsphere computations, the value
    assigned to a simplex is often the squared radius of its circumsphere
    restricted to the simplex dimension, with additional handling for obtuse
    simplices in low dimensions. The exact convention must match the algorithm
    that filled ``alpha_sq``.
    """

    alpha_sq: Dict[Simplex, float]