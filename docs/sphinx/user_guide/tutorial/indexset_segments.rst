.. ##
.. ## Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _indexset-label:

-----------------------------------------
Iteration Spaces: IndexSets and Segments
-----------------------------------------

Key RAJA features shown in this example are:

  * ``RAJA::forall`` loop execution template
  * ``RAJA::RangeSegment`` (i.e., ``RAJA::TypedRangeSegment``) iteration space construct
  * ``RAJA::TypedListSegment`` iteration space construct
  * ``RAJA::IndexSet`` iteration construct and associated execution policies

The file ``RAJA/examples/tut_indexset-segments.cpp`` contains working code
for the examples discussed in this section.

The example uses a simple daxpy kernel and its usage of RAJA is similar to
previous simple loop examples. 

.. literalinclude:: ../../../../examples/tut_indexset-segments.cpp
   :start-after: _csytle_daxpy_start
   :end-before: _csytle_daxpy_end
   :language: C++

The examples in this section focuse on how to use RAJA index sets and iteration 
space segments, such as index ranges and lists of indices. Lists of indices 
are important for algorithms that use indirection arrays for irregular array 
accesses. Combining different segment types, such as ranges and lists in an 
index set allows a user to launch different iteration patterns in a single loop 
execution construct (i.e., one kernel). This is something that is not 
supported by other programming models and abstractions and is unique to RAJA. 
Applying these concepts judiciously can increase performance by allowing 
compilers to optimize for specific segment types (e.g., SIMD for range 
segments) while providing the flexibility of indirection arrays for general
indexing patterns.

.. note:: For the following examples, it is useful to remember that all
          RAJA segment types are templates, where the type of the index
          value is the template argument. So for example, the basic RAJA
          range segment type is ``RAJA::TypedRangeSegment<T>``. The type
          ``RAJA::RangeSegment`` used here (for convenience) is a type alias 
          for ``RAJA::TypedRangeSegment<RAJA::Index_type>``, where the
          template parameter is a default index type that RAJA defines.

For a summary discussion of RAJA segment and index set concepts, please 
see :ref:`index-label`.

^^^^^^^^^^^^^^^^^^^^^
RAJA Segments
^^^^^^^^^^^^^^^^^^^^^

In previous examples, we have seen how to define a contiguous range of loop
indices [0, N) with a ``RAJA::RangeSegment(0, N)`` object and use it in a RAJA
loop execution template to run a loop kernel over the range. For example:

.. literalinclude:: ../../../../examples/tut_indexset-segments.cpp
   :start-after: _rajaseq_daxpy_range_start
   :end-before: _rajaseq_daxpy_range_end
   :language: C++

We can accomplish the same result by enumerating the indices in a 
``RAJA::TypedListSegment`` object. Here, we assemble the indices in a standard 
vector, create a list segment from that, and then pass the list segment to the 
forall execution template:

.. literalinclude:: ../../../../examples/tut_indexset-segments.cpp
   :start-after: _rajaseq_daxpy_list_start
   :end-before: _rajaseq_daxpy_list_end
   :language: C++

Note that we are using the following type aliases:

.. literalinclude:: ../../../../examples/tut_indexset-segments.cpp
   :start-after: _raja_list_segment_type_start
   :end-before: _raja_list_segment_type_end
   :language: C++

Recall from discussion in :ref:`index-label` that ``RAJA::Index_type`` is
a default index type that RAJA defines and which is used in some RAJA
constructs as a convenience for users who want a simple mechanism to apply
index types consistently.

It is important to understand what happens when using list segments.
During loop execution, indices stored in the list segment are passed to the 
loop body one-by-one, effectively mimicking an indirection array except that
the indirection does not appear explicitly in the array indexing in the loop 
body. 

Another simple example helps to reinforce the list segment concept.
We can reverse the order of the indices, run the loop backwards with 
a new list segment object:

.. literalinclude:: ../../../../examples/tut_indexset-segments.cpp
   :start-after: _raja_list_segment_daxpy_reverse_start
   :end-before: _raja_list_segment_daxpy_reverse_end
   :language: C++

Alternatively, we can also use a RAJA strided range segment to run the loop 
in reverse by giving it a stride of -1. For example:

.. literalinclude:: ../../../../examples/tut_indexset-segments.cpp
   :start-after: _raja_range_segment_daxpy_negstride_start
   :end-before: _raja_range_segment_daxpy_negstride_end
   :language: C++

The fact that RAJA always passes loop index values to a lambda kernel
explains why we can run a kernel with different segment types in a single,
portable RAJA-based kernel. Try doing this sort of thing with straight C code.

^^^^^^^^^^^^^^^^^^^^^
RAJA IndexSets
^^^^^^^^^^^^^^^^^^^^^

The ``RAJA::TypedIndexSet`` template is a container that can hold
any number of segments, of the same or different types. An index set object 
can be passed to a RAJA loop execution method, just like a segment, to
run a loop kernel. When the loop is run, the execution method iterates 
over the segments and the loop indices in each segment. Thus, the loop 
iterates can be grouped into different segments to partition the iteration 
space and iterate over the loop kernel chunks (defined by segments), in 
serial, in parallel, or in some specific dependency ordering. Individual 
segments can be executed in serial or parallel.

When an index set is defined, the segment types it may hold must be specified
as template arguments. For example, here we create an index set that can
hold list segments. Then, we add the list segment we created earlier to it
and run the loop:

.. literalinclude:: ../../../../examples/tut_indexset-segments.cpp
   :start-after: _raja_indexset_list_daxpy_start
   :end-before: _raja_indexset_list_daxpy_end
   :language: C++

You are probably wondering: What is the 'SEQ_ISET_EXECPOL' type used for the 
execution policy? 

Well, it is similar to execution policy types we have seen up to this point, 
except that it specifies a two-level policy -- one for iterating over the 
segments and one for executing the iterates defined by each segment. In the 
example, we specify that we should do each of these operations sequentially 
by defining the policy as follows:

.. literalinclude:: ../../../../examples/tut_indexset-segments.cpp
   :start-after: _raja_seq_indexset_policy_daxpy_start
   :end-before: _raja_seq_indexset_policy_daxpy_end
   :language: C++

Next, we perform the daxpy operation by partitioning the iteration space into
two range segments:

.. literalinclude:: ../../../../examples/tut_indexset-segments.cpp
   :start-after: _raja_indexset_2ranges_daxpy_start
   :end-before: _raja_indexset_2ranges_daxpy_end
   :language: C++

The first range segment is used to run the index range [0, N/2) and the
second is used to run the range [N/2, N).

We can also break up the iteration space into three segments, 2 ranges 
and 1 list:

.. literalinclude:: ../../../../examples/tut_indexset-segments.cpp
   :start-after: _raja_indexset_2ranges_1list_daxpy_start 
   :end-before: _raja_indexset_2ranges_1list_daxpy_end
   :language: C++

The first range segment runs the index range [0, N/3), the list segment
enumerates the indices in the interval [N/3, 2*N/3), and the second range
segment runs the range [2*N/3, N). Note that we use the same execution
policy as before. 

Before we end the discussion of these examples, we demonstrate a few more 
index set execution policy variations. To run the previous three segment 
code by iterating over the segments sequentially and executing each 
segment in parallel using OpenMP multithreading, we would use this policy 
definition:

.. literalinclude:: ../../../../examples/tut_indexset-segments.cpp
   :start-after: _raja_indexset_ompinnerpolicy_daxpy_start
   :end-before: _raja_indexset_ompinnerpolicy_daxpy_end
   :language: C++

If we wanted to iterate over the segments in parallel using OpenMP 
multi-threading and execute each segment sequentially, we would use the
following policy:

.. literalinclude:: ../../../../examples/tut_indexset-segments.cpp
   :start-after: _raja_indexset_ompouterpolicy_daxpy_start
   :end-before: _raja_indexset_ompouterpolicy_daxpy_end
   :language: C++ 

Finally, to iterate over the segments sequentially and execute each segment in
parallel on a GPU using either CUDA or HIP kernel, we would use a policy,
such as:
 
.. literalinclude:: ../../../../examples/tut_indexset-segments.cpp
   :start-after: _raja_indexset_cudapolicy_daxpy_start
   :end-before: _raja_indexset_cudapolicy_daxpy_end
   :language: C++

or:

.. literalinclude:: ../../../../examples/tut_indexset-segments.cpp
   :start-after: _raja_indexset_hippolicy_daxpy_start
   :end-before: _raja_indexset_hippolicy_daxpy_end
   :language: C++

