.. _view::
.. _ref-view:

===============
View and Layout
===============

To simplify multi-dimensional indexing, RAJA introduces the ``RAJA::View``.
The basic usage of the ``RAJA::View`` is as follows

.. code-block:: cpp

   RAJA::View<double, RAJA::Layout<DIM>> Aview(Aptr, N1, ..., Nn);

Here ``Aptr`` corresponds to a pointer. The ``RAJA::View`` is templated
on a type (ex. double, float, int, ...), and ``N1, ... , Nn``
identifies the stride in each dimension. 

The ``RAJA::Layout<DIM>`` encapsulates the number of dimensions , ``DIM`` , the ``RAJA::View``
will have.



