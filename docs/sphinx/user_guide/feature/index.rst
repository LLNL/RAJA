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

.. _index-label:

====================
Indices and Segments
====================

-------
Indices
-------

The recommended loop counter for RAJA is the ``RAJA::Index_type``; a 64-bit loop counter which makes it
easy for the compiler to carryout optimizations. Basic usage may be found in ``examples-add-vectors.cpp``.

--------
Segments
--------

Segments correspond to the iteration space that will be traversed by a RAJA loop. The following is a catalog of containers
included in RAJA.

* ``RAJA::RangeSegment(start, stop)`` - Generates a contiguous sequence of numbers starting from start but not including stop

* ``RAJA::ListSegment`` - A container which holds ``RAJA::Index_types``

* ``RAJA::TypedListSegment<data_type>`` - A general purpose container which holds a specified data_type (ex. int, long int)

* ``RAJA::StaticIndexSet<>``  - Holds a collection of ``RAJA::ListSegments`` or ``RAJA::TypedListSegments``

Basic usage may be found in ``examples-add-vectors.cpp`` and ``example-gauss-sidel.cpp``
