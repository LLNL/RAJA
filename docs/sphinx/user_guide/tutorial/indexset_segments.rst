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

.. _indexset-label:

----------------------
IndexSets and Segments
----------------------

Key RAJA features shown in this example:

  * ``RAJA::forall`` loop traversal template
  * ``RAJA::RangeSegment`` iteration space construct
  * ``RAJA::ListSegment`` iteration space construct
  * ``RAJA::IndexSet`` iteration construct and associated execution policies

The example re-uses the daxpy kernel from an earlier example. It focuses 
on how to use RAJA index sets and iteration space segments. These features
are important for applications and algorithms that need to use indirection 
arrays for irregular array accesses. Combining different segment types, such
as ranges and lists in an index set allows users to launch different iteration
patterns in a single loop execution construct (i.e., one kernel). This is 
something that is not supported by other programming models and abstractions
and is unique to RAJA. Using these constructs can increase performance by 
allowing compilers to optimize for specific segment types (e.g., SIMD for 
range segments).

The file ``RAJA/examples/ex3-indexset-segments.cpp`` contains the complete 
working example code.
