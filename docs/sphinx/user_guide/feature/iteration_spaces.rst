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

Loop variables and their associated iteration spaces are fundamental to 
writing loop kernels in RAJA. RAJA provides some basic iteration space types 
that serve as flexible building blocks that can be used to form a variety 
of loop iteration patterns. For example, these types can be used to 
aggregate, partition, (re)order, etc. a set of loop iterates. In this
section, we introduce RAJA index and iteration space concepts and types.

More examples of RAJA iteration space usage can be found in the
:ref:`indexset-label` and :ref:`vertexsum-label` sections of the tutorial.

.. note:: All RAJA iteration space types described here are located in the 
          namespace ``RAJA``.

.. _indices-label:

-------
Indices
-------

Just like traditional C and C++ for-loops, RAJA uses index variables to 
identify loop iterates; e.g., a lambda expression loop body takes an index
variable argument. RAJA containers and methods are templated in
a sufficiently general way to allow users to use any integral type for an
index variable. In most cases, the index variable type is explicitly defined
by users. However, RAJA also provides a ``RAJA::Index_type`` type, which is 
used as a default in some circumstances for convenience. For example, this
allows use of common aliases to typed constructs without specifying the type. 
The ``RAJA::Index_type`` type is an alias to the C++ type 'std::ptrdiff_t', 
which is appropriate for most compilers to generate useful loop-level 
optimizations.

Users can change the type of ``RAJA::Index_type`` by editing the RAJA
``RAJA/include/RAJA/util/types.hpp`` header file.

.. _segments-label:

-------------
Segments
-------------

A RAJA 'Segment' is a container of loop iterates that one wants to 
execute as a unit. RAJA provides Segment types for contiguous index ranges, 
constant (non-unit) stride ranges, and arbitrary lists of indices.

Stride-1 Segments
^^^^^^^^^^^^^^^^^^^

A ``RAJA::TypedRangeSegment`` is the fundamental type for representing a 
stride-1 (i.e., contiguous) range of indices.

.. figure:: ../figures/RangeSegment.png

   A range segment defines a stride-1 index range [beg, end).

One can create an explicitly-typed range segment or one with the default
``RAJA::Index_type`` index type. For example,::

   // A stride-1 index range [beg, end) of type int.
   RAJA::TypedRangeSegment<int> int_range(beg, end);

   // A stride-1 index range [beg, end) of the RAJA::Index_type default type
   RAJA::RangeSegment default_range(beg, end);

Strided Segments
^^^^^^^^^^^^^^^^^^^

A ``RAJA::TypedRangeStrideSegment`` defines a range with a constant stride
that is given explicitly stride, including negative stride.

.. figure:: ../figures/RangeStrideSegment.png

   A range-stride segment defines an index range with arbitrary stride [beg, end, stride).

One can create an explicitly-typed strided range segment or one with the 
default ``RAJA::Index_type`` index type. For example,::

   // A stride-2 index range [beg, end, 2) of type int.
   RAJA::TypedRangeStrideSegment<int> stride2_range(beg, end, 2);

   // A index range with -1 stride [0, N-1, -1) of the RAJA::Index_type default type
   RAJA::RangeStrideSegment neg1_range( N-1, -1, -1);

Using a negative stride range in a RAJA loop traversal template will run the
indices in revers order. For example, using 'neg1_range' from above::

   RAJA::forall< RAJA::seq_exec >( neg1_range, [=] (RAJA::Index_type i) {
     printf("%ld ", i); 
   } );

will print the values::

   N-1  N-2  N-3 .... 1 0 

RAJA strided ranges support both positive and negative stride values. The
following items are worth noting:

.. note:: When using a RAJA strided range, no loop iterations will be run
          under the following conditions:
            * Stride > 0 and begin > end
            * Stride < 0 and begin < end
            * Stride == 0

List Segments
^^^^^^^^^^^^^^

A ``RAJA::TypedListSegment`` is used to define an arbitrary set of loop 
indices, akin to an indirection array.

.. figure:: ../figures/ListSegment.png

   A list segment defines an arbitrary collection of indices. Here, we have a list segment with 5 irregularly-spaced indices.

A list segment is created by passing an array of integral values to its
constructor. For example::

   // Create a vector holding some integer index values
   std::vector<int> idx = {0, 2, 3, 4, 7, 8, 9, 53};

   // Create list segment with these loop indices
   RAJA::TypedListSegment<int> idx_list( &idx[0], static_cast<int>(idx.size()) );

Similar to range segment types, RAJA provides ``RAJA::ListSegment``, which is
a type alias to ``RAJA::TypedListSegment`` using ``RAJA::Index_type`` as the
template type parameter.
   
Segment Types and  Iteration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is worth noting that RAJA segment types model **C++ iterable interfaces**.
In particular, each segment type defines three methods:

  * begin()
  * end()
  * size()

and two types:

  * iterator (essentially a *random access* iterator type)
  * value_type

Thus, any iterable type that defines these methods and types appropriately
can be used as a segment with RAJA traversal templates.

.. _indexsets-label:

--------------
IndexSets
--------------

A ``RAJA::TypedIndexSet`` is a container that can hold an arbitrary collection
of segment objects of arbitrary type as illustrated in the following figure.

.. figure:: ../figures/IndexSet.png

   An index set with 2 range segments and one list segment.

We can create an index set that describes an iteration space like this as
follows::

   // Create an index set that can hold range and list segments with the
   // default index type
   RAJA::TypedIndexSet< RAJA::RangeSegment, RAJA::ListSegment > iset;

   // Add two range segments and one list segment to the index set
   iset.push_back( RAJA::RangeSegment( ... ) );
   iset.push_back( RAJA::ListSegment(...) );
   iset.push_back( RAJA::RangeSegment( ... ) );

Now that we've created this index set object, we can pass it to any RAJA 
loop execution template to execute the indices defined by all of its segments::

   // Define an index set execution policy type that will iterate over
   // its segments in parallel (OpenMP) and execute each segment sequentially 
   using ISET_EXECPOL = RAJA::ExecPolicy< RAJA::omp_parallel_segit, 
                                          RAJA::seq_exec >;

   // Run a kernel with iterates defined by the index set
   RAJA::forall<ISET_EXECPOL>(iset, [=] (int i) { ... });

The loop iterations will execute in three chunks defined by the two range 
segments and one list segment. The segments will be iterated over in
parallel using OpenMP, and each segment will execute sequentially.
