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

.. _indexset-label:

-----------------------------------------
Iteration Spaces: IndexSets and Segments
-----------------------------------------

Key RAJA features shown in this example:

  * ``RAJA::forall`` loop traversal template
  * ``RAJA::RangeSegment`` (i.e., ``RAJA::TypedRangeSegment``) iteration space construct
  * ``RAJA::TypedListSegment`` iteration space construct
  * ``RAJA::IndexSet`` iteration construct and associated execution policies

The example re-uses the daxpy kernel from an earlier example. It focuses 
on how to use RAJA index sets and iteration space segments, such as index 
ranges and lists of indices. These features are important for applications 
and algorithms that use indirection arrays for irregular array accesses. 
Combining different segment types, such as ranges and lists in an index set 
allows a user to launch different iteration patterns in a single loop 
execution construct (i.e., one kernel). This is something that is not 
supported by other programming models and abstractions and is unique to RAJA. 
Applying these concepts judiciously can increase performance by allowing 
compilers to optimize for specific segment types (e.g., SIMD for range 
segments) while providing the flexibility of indirection arrays.

.. note:: Any iterable type that provides the methods 'begin', 'end', and
          'size' and an 'iterator' type and 'value_type' type can be used
          as a *segment* with RAJA traversal templates.

^^^^^^^^^^^^^^^^^^^^^
RAJA RangeSegments
^^^^^^^^^^^^^^^^^^^^^

In earlier examples, we have seen how to specify a contiguous range of loop
indices [0, N) using a ``RAJA::RangeSegment`` object. For example:

.. literalinclude:: ../../../../examples/ex3-indexset-segments.cpp
                    :lines: 121-123

.. note:: All RAJA Segment types are templated on the type of their index 
          values.

For example, ``RAJA::RangeSegment`` is actually a type alias to 
``RAJA::TypedRangeSegment<RAJA::Index_type>``. The ``RAJA::Index_type`` is 
defined to be ``std::ptrdiff_t`` by default, which seems to be best for most 
compilers to generate loop optimizations, such as SIMD. 

Users can change the type of ``RAJA::Index_type`` by editing the RAJA 
``types.hpp`` header file.

^^^^^^^^^^^^^^^^^^^^^
RAJA ListSegments
^^^^^^^^^^^^^^^^^^^^^

We can accomplish the same result by enumerating the indices in a 
``RAJA::TypedListSegment`` object. Here, we add the indices to a standard 
vector, create a list segment from it, and then pass the list segment to the 
forall template:

.. literalinclude:: ../../../../examples/ex3-indexset-segments.cpp
		    :lines: 139-148

Note that we are using the following type aliases:

.. literalinclude:: ../../../../examples/ex3-indexset-segments.cpp
		    :lines: 57-58

It is important to note what is really going on here. During the loop
execution, the indices stored in the list segment are passed to the loop
body one-by-one, effectively mimicking an indirection array except that
the indirection array does not appear in the loop body. This allows loops
to be run with RAJA using any properly defined segment type, including 
user-defined types. For example, we can reverse the order of the indices,
run the loop with a new list segment object, and get the same result since
the loop is `data-parallel`:

.. literalinclude:: ../../../../examples/ex3-indexset-segments.cpp
                    :lines: 166-170

^^^^^^^^^^^^^^^^^^^^^
RAJA IndexSets
^^^^^^^^^^^^^^^^^^^^^

The ``RAJA::TypedIndexSet`` template is a container that holds segment
types specified as template arguments. An index set object can be passed
to the RAJA 'forall' method just like a segment. When the loop is run,
the traversal will iterate over the segments and each segment will be 
executed. Here, we create an index set, add the first list segment from 
above to it, and run the loop:

.. literalinclude:: ../../../../examples/ex3-indexset-segments.cpp
                    :lines: 192-198

What is the 'SEQ_ISET_EXECPOL' type used for the execution policy? 

Well, it is like the execution policy types we have seen up to this point, 
except that it specifies a two-level policy -- one for iterating over the 
segments and one for executing each segment. In the example, we specify
that we should do each of these operations sequentially by defining the
policy as follows:

.. literalinclude:: ../../../../examples/ex3-indexset-segments.cpp
                    :lines: 183-184

Next, we perform the daxpy operation by partitioning the iteration space into
two range segments:

.. literalinclude:: ../../../../examples/ex3-indexset-segments.cpp
                    :lines: 210-216

The first range segment is used to run the index range [0, N/2) and the
second is used to run the range [N/2, N).

We could also break up the iteration space into three segments, 2 ranges 
and 1 list:

.. literalinclude:: ../../../../examples/ex3-indexset-segments.cpp
                    :lines: 231-245

The first range segment runs the index range [0, N/3), the list segment
enumerates the indices in the interval [N/3, 2*N/3), and the second range
segment runs the range [2*N/3, N). Note that we use the same execution
policy as before. 

Before we close the discussion on this topic, we demonstrate a few more 
execution policy variations. To run the previous three segment example
by iterating over the segments sequentially and execute each segment in
parallel using OpenMP multi-threading, we would use this policy definition:

.. literalinclude:: ../../../../examples/ex3-indexset-segments.cpp
                    :lines: 264-265

If we wanted to iterate over the segements in parallel using OpenMP 
multi-threading and execute each segment sequentially, this is the policy
we would define:

.. literalinclude:: ../../../../examples/ex3-indexset-segments.cpp
                    :lines: 283-284

Finally, to iterate over the segments sequentially and execute each segment in
parallel on a GPU by launching a CUDA kernel, we would define this policy:
 
.. literalinclude:: ../../../../examples/ex3-indexset-segments.cpp
                    :lines: 301-302

The file ``RAJA/examples/ex3-indexset-segments.cpp`` contains the complete 
working example code.
