"""
Simplex utilities: canonicalization, dimension, and face iteration.

This example demonstrates three basic utilities from :mod:`homolipop.simplices`.

- :func:`homolipop.simplices.canonical_simplex` for canonicalizing a vertex tuple
- :func:`homolipop.simplices.simplex_dim` for computing the simplex dimension
- :func:`homolipop.simplices.iter_faces` for enumerating faces of a given dimension

Mathematical conventions
========================

Canonical simplex representation
--------------------------------

A simplex is represented by a tuple of vertex indices.

Given an arbitrary finite tuple of integers

.. math::

   v = (v_0,\\dots,v_m),

we interpret it as specifying a set of vertices.
The canonical simplex associated to :math:`v` is the strictly increasing tuple obtained by

1. removing duplicate entries
2. sorting the remaining vertices in increasing order

Equivalently, if :math:`S = \\{v_0,\\dots,v_m\\}` is the set of values appearing in :math:`v`,
then

.. math::

   \\mathrm{canonical\\_simplex}(v) = (w_0,\\dots,w_k),

where :math:`w_0 < \\cdots < w_k` are the elements of :math:`S` in increasing order.

Dimension
---------

If :math:`\\sigma = (w_0,\\dots,w_k)` is a canonical simplex with :math:`k+1` vertices,
its dimension is

.. math::

   \\dim(\\sigma) = k.

Face enumeration
----------------

Let :math:`\\sigma` be a simplex with vertex set :math:`V(\\sigma)`.
For :math:`r \\in \\{0,\\dots,\\dim(\\sigma)\\}`, an :math:`r` face of :math:`\\sigma`
is a simplex obtained by choosing :math:`r+1` vertices from :math:`V(\\sigma)`.

The generator :func:`homolipop.simplices.iter_faces` is assumed to yield all faces of
a prescribed dimension ``face_dim`` as canonical tuples, in a deterministic order.

In this example, the canonicalization step maps ``raw = (5, 2, 7, 2)`` to the simplex
``(2, 5, 7)``, which is a 2-simplex. The printed messages refer to a 3-simplex only if
the input has four distinct vertices. With the given input, the simplex is a triangle,
so its 2-faces are itself and it has no 3-dimensional structure.

To avoid confusion, this script prints faces relative to the actual computed dimension.
"""

from __future__ import annotations

from homolipop.simplices import canonical_simplex, iter_faces, simplex_dim


def main() -> None:
    """
    Canonicalize a raw vertex tuple, compute its dimension, and enumerate faces.

    The example input contains duplicates. After canonicalization, the simplex has
    distinct, sorted vertices. The script then prints all faces of dimensions 2 and 1
    that exist for the resulting simplex.
    """
    raw = (5, 2, 7, 2)
    simplex = canonical_simplex(raw)

    dim = simplex_dim(simplex)

    print("Raw vertices:", raw)
    print("Canonical simplex:", simplex)
    print("Dimension:", dim)

    if dim >= 2:
        print(f"All 2-faces of the {dim}-simplex:")
        for face in iter_faces(simplex, face_dim=2):
            print(face)
    else:
        print("No 2-faces exist, because the simplex dimension is < 2.")

    if dim >= 1:
        print(f"All 1-faces of the {dim}-simplex:")
        for face in iter_faces(simplex, face_dim=1):
            print(face)
    else:
        print("No 1-faces exist, because the simplex dimension is < 1.")


if __name__ == "__main__":
    main()