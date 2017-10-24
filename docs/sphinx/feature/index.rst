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

.. _index::
.. _ref-index:

.. # ====================
.. # Indices and Segments
.. # ====================

-------
Indices 
-------

RAJA introduces a custom 64-bit loop counter

* ``RAJA::Index_type``

designed to make it easy for the compiler to perform optimization on. Basic usage of may be found in ``examples-add-vectors.cpp``.

--------
Segments
--------

RAJA also includes various index containers.

* ``RAJA::RangeSegment(start, stop)`` - Generates a contingous sequence of numbers

* ``RAJA::ListSegment`` - A general purpose container which holds a specified data_type

* ``RAJA::TypedListSegment<data_type>`` - A general purpose container which holds a specified data_type (ex. int, long int)

* ``RAJA::StaticIndexSet<>``  - Holds a collection of RAJA::ListSegments or RAJA::TypedListSegments

Basic usage may be found in ``examples-add-vectors.cpp`` and ``example-gauss-sidel.cpp``
