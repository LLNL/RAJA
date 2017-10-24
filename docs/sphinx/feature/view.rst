.. ##
.. ## Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
.. ##
.. ## Produced at the Lawrence Livermore National Laboratory
.. ##
.. ## LLNL-CODE-689114
.. ##
.. ## All rights reserved.
.. ##
.. ## This file is part of RAJA.
.. ##
.. ## For details about use and distribution, please read RAJA/LICENSE.
.. ##

.. _view-label:

===============
View and Layout
===============

To simplify multi-dimensional indexing, RAJA introduces the ``RAJA::View``.
The basic usage of the ``RAJA::View`` is as follows::

   RAJA::View<double, RAJA::Layout<DIM>> Aview(Aptr, N1, ..., Nn);

Here ``Aptr`` corresponds to a pointer. The ``RAJA::View`` is templated
on a type (ex. double, float, int, ...), and ``N1, ... , Nn``
identifies the stride in each dimension. 

The ``RAJA::Layout<DIM>`` encapsulates the number of dimensions , ``DIM`` , the ``RAJA::View``
will have.



