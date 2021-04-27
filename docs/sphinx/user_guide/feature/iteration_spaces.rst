.. ##
.. ## Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
.. ## and other RAJA project contributors. See the RAJA/COPYRIGHT file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _index-label:

================================
Indices, Segments, and IndexSets
================================

Loop variables and their associated iteration spaces are fundamental to 
writing loop kernels in RAJA. RAJA provides some basic iteration space types 
that serve as flexible building blocks that can be used to form a variety 
of loop iteration patterns. These types can be used to define a particular 
order for loop iterates, aggregate and partition iterates, as well as other
configurations. In this section, we introduce RAJA index and iteration space 
concepts and types.

More examples of RAJA iteration space usage can be found in the
:ref:`indexset-label` and :ref:`vertexsum-label` sections of the tutorial.

.. note:: All RAJA iteration space types described here are located in the 
          namespace ``RAJA``.

.. _indices-label:

-------
Indices
-------

Just like traditional C and C++ for-loops, RAJA uses index variables to 
identify loop iterates. Any lambda expression that represents all or part of
a loop body passed to a ``RAJA::forall`` or ``RAJA::kernel`` method will 
take at least one loop index variable argument. RAJA iteration space types 
are templates that allow users to use any integral type for an
index variable. The index variable type may be explicitly specified by a user.
RAJA also provides the ``RAJA::Index_type`` type, which is used as a default 
in some circumstances for convenience by allowing use of a common type 
alias to typed constructs without explicitly specifying the type. 
The ``RAJA::Index_type`` type is an alias to the C++ type ``std::ptrdiff_t``, 
which is appropriate for most compilers to generate useful loop-level 
optimizations.

.. _segments-label:

-------------
Segments
-------------

A RAJA **Segment** represents a set of loop indices that one wants to 
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

   // A stride-1 index range [beg, end) using type int.
   RAJA::TypedRangeSegment<int> int_range(beg, end);

   // A stride-1 index range [beg, end) using the RAJA::Index_type default type
   RAJA::RangeSegment default_range(beg, end);

.. note:: When using a RAJA range segment, no loop iterations will be run when
          begin is greater-than-or-equal-to end similar to a C-style for-loop.

Strided Segments
^^^^^^^^^^^^^^^^^^^

A ``RAJA::TypedRangeStrideSegment`` defines a range with a constant stride
that is given explicitly stride, including negative stride.

.. figure:: ../figures/RangeStrideSegment.png

   A range-stride segment defines an index range with arbitrary stride [beg, end, stride).

One can create an explicitly-typed strided range segment or one with the 
default ``RAJA::Index_type`` index type. For example,::

   // A stride-2 index range [beg, end, 2) using type int.
   RAJA::TypedRangeStrideSegment<int> stride2_range(beg, end, 2);

   // A index range with -1 stride [0, N-1, -1) using the RAJA::Index_type default type
   RAJA::RangeStrideSegment neg1_range( N-1, -1, -1);

Using a range with a stride  of '-1' as above in a RAJA loop traversal template
will run the loop indices in reverse order. That is, using 'neg1_range' 
from above::

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

A list segment is created by passing an array of integral values to a list
segment constructor. For example::

   // Create a vector holding some integer index values
   std::vector<int> idx = {0, 2, 3, 4, 7, 8, 9, 53};

   // Create list segment with these loop indices where the indices are 
   // stored in the host memory space
   camp::resources::Resource host_res{camp::resources::Host()};
   RAJA::TypedListSegment<int> idx_list( &idx[0], idx.size(),
                                         host_res );

Using a list segment in a RAJA loop traversal template will run the loop 
indices specified in the array passed to the list segment constructor. That 
is, using 'idx_list' from above::

   RAJA::forall< RAJA::seq_exec >( idx_list, [=] (RAJA::Index_type i) {
     printf("%ld ", i);
   } );

will print the values::

   0 2 3 4 7 8 9 53

Note that a ``RAJA::TypedListSegment`` constructor can take a pointer to
an array of indices and an array length, as shown above. If the indices are
in a container, such as ``std::vector`` that provides ``begin()``, ``end()``,
and ``size()`` methods, the length argument is not required. For example::

   std::vector<int> idx = {0, 2, 3, 4, 7, 8, 9, 53};

   camp::resources::Resource host_res{camp::resources::Host()};
   RAJA::TypedListSegment<int> idx_list( idx, host_res );

Similar to range segment types, RAJA provides ``RAJA::ListSegment``, which is
a type alias to ``RAJA::TypedListSegment`` using ``RAJA::Index_type`` as the
template type parameter.
   
By default, the list segment constructor copies the indices in the array
passed to it to the memory space specified by the resource argument.
The resource argument is required so that the segment index values are in the
proper memory space for the kernel to run. Since the kernel is run on 
the CPU host in this example (indicated by the ``RAJA::seq_exec`` execution 
policy), we pass a host resource object to the list segment constructor. 
If, for example, the kernel was to run on a GPU using a CUDA or HIP 
execution policy, then the resource type passed to the camp resource 
constructor would be ``camp::resources::Cuda()`` or 
``camp::resources::Hip()``, respectively.

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
of segment objects of arbitrary type as illustrated in the following figure,
where we have two contiguous ranges and an irregularly-spaced list of indices.

.. figure:: ../figures/IndexSet.png

   An index set with 2 range segments and one list segment.

We can create an index set that describes such an iteration space::

   // Create an index set that can hold range and list segments with the
   // default index type
   RAJA::TypedIndexSet< RAJA::RangeSegment, RAJA::ListSegment > iset;

   // Add two range segments and one list segment to the index set
   iset.push_back( RAJA::RangeSegment( ... ) );
   iset.push_back( RAJA::ListSegment(...) );
   iset.push_back( RAJA::RangeSegment( ... ) );

Now that we've created this index set object, we can pass it to any RAJA 
loop execution template to execute the indices defined by its segments::

   // Define an index set execution policy type that will iterate over
   // its segments in parallel (OpenMP) and execute each segment sequentially 
   using ISET_EXECPOL = RAJA::ExecPolicy< RAJA::omp_parallel_segit, 
                                          RAJA::seq_exec >;

   // Run a kernel with iterates defined by the index set
   RAJA::forall<ISET_EXECPOL>(iset, [=] (int i) { ... });

In this example, the loop iterations will execute in three chunks defined by 
the two range segments and one list segment. The segments will be iterated 
over in parallel using OpenMP, and each segment will execute sequentially.

.. note:: Iterating over the indices of all segments in a RAJA index set 
          requires a two-level execution policy, with two template parameters,
          as shown above. The first parameter specifies how to iterate over 
          the seqments. The second parameter specifies how each segment will 
          execute. See :ref:`indexsetpolicy-label` for more information about 
          RAJA index set execution policies.

.. note:: It is the responsibility of the user to ensure that segments are
          defined properly when using RAJA index sets. For example, if the
          same index appears in multiple segments, the corresponding loop
          iteration will be run multiple times.
