.. ##
.. ## Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
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

================================
Indices, Segments, and IndexSets
================================

Loop variables and iteration spaces are fundamental to C/C++ loops. In this 
section, we introduce RAJA specific index types and iteration spaces which 
are designed to encapsulate common programming patterns.

.. note:: All types described here are located in the RAJA namespace.

-------
Indices
-------

All RAJA index segments may use any loop variable type. The 'Typed' segments 
that RAJA provides are templated on a loop variable type. RAJA also provides
a ``RAJA::Index_type`` type that is an alias to a 64-bit integer variable,
which is appropriate for most compilers to generate optimizations. RAJA also
provides (non-typed) aliases for segments that use ``RAJA::Index_type`` by 
default.

-------------
RAJA Segments
-------------

In RAJA, a 'Segment' is a portion of a loop iteration space that you would
want to execute as a unit. RAJA provides a few important Segment types for
contiguous index ranges, strided ranges, and arbitrary lists of indices.
For example,::

   // A contiguous index range [start, stop) 
   RAJA::TypedRangeSegment<T>(T start, T stop)  

   // A contiguous index range using the default index type
   RAJA::RangeSegment loopBounds(RAJA::Index_type start, RAJA::Index_type stop)

   // A strided index range
   RAJA::TypedRangeStrideSegment<T>(T start, T stop, T stride)

.. note:: * TypedRangeStrideSegment::iterator is a random access iterator
          * TypedRangeStrideSegment allows for positive or negative strides, but a stride of zero is undefined
          * For positive strides, begin() > end() implies size()==0
          * For negative strides, begin() < end() implies size()==0

A RAJA list segment can be used to iterate over any index sequence. For 
example::

    // Aptr points to an array of values and Alen is number of values
    RAJA::TypedListSegment<T>(T *Aptr, size_t Alen)

    // A list segment using the default index type
    RAJA::ListSegment(RAJA::Index_type *Aptr,size_t Alen)

.. note:: Any iterable type that defines methods 'begin()', 'end()', and 
          'size()' and 'iterator' and 'value_type' types can be used as a 
          segment with RAJA traversal templates.

-------------
RAJA Segments
-------------

A ``RAJA::TypedIndexSet`` is a segment variadic template container templated 
on the types of segments it may hold. It may hold any number of segments of
each specified type. An IndexSet object can be passed to a RAJA traversal 
template to execute all of its segments. For example,::

  RAJA::TypedIndexSet< RAJA::RangeSegment, RAJA::TypedListSegment<int> > iset;

  iset.push_back( RAJA::RangeSegment( ... ) );
  iset.push_back( RAJA::TypedListSegment<int>(...) );
  iset.push_back( RAJA::RangeSegment( ... ) );

  using ISET_EXECPOL = RAJA::ExecPolicy< RAJA::omp_parallel_segit, 
                                         RAJA::seq_exec >;

  RAJA::forall<ISET_EXECPOL>(iset, [=] (int i) { ... });

will execute iterations for a given loop in three chunks defined by the two
range segments and one list segment. The segments will be iterated over in
parallel using OpenMP, and each segment will execute sequentially.

For more information, please see the :ref:`indexset-label` tutorial section.
