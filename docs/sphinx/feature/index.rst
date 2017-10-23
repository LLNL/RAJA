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

====================
Indices and Segments
====================
Here we provide an overview of loop counters and iteratation spaces included with RAJA. 

-------
Indices 
-------

Although RAJA is capable of taking any type of loop counter, the recommend type is a ``RAJA::Index_type``. 

1. RAJA::Index_type - 32-bit loop counter. 

--------
Segments
--------
RAJA also includes various index containers. Here we catalog containers included in RAJA. 

1. RAJA::RangeSegment(start, stop) - Generates a contingous sequence of numbers

2. RAJA::ListSegment - A general purpose container which holds a specified data_type

3. RAJA::TypedListSegment<data_type> - A general purpose container which holds a specified data_type (ex. int, long int)

4. RAJA::StaticIndexSet<>  - Holds a collection of RAJA::ListSegments or RAJA::TypedListSegments
