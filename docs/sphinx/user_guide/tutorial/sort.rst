.. ##
.. ## Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/COPYRIGHT file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _sort-label:

--------------------------------------------------
Parallel Sort Operations
--------------------------------------------------

Key RAJA features shown in this section:

  * ``RAJA::sort`` operation
  * ``RAJA::sort_pairs`` operation
  * ``RAJA::stable_sort`` operation
  * ``RAJA::stable_sort_pairs`` operation
  * RAJA comparators for different types of sorts; e.g., less, greater

Below, we present examples of RAJA sequential, OpenMP,
and CUDA sort operations and show how different sort orderings can be
achieved by passing different RAJA comparators to the RAJA sort template
methods. Each comparator is a template type, where the template argument is
the type of the values it compares. For a summary of RAJA sort
functionality, please see :ref:`sort-label`.

.. note:: RAJA sort operations use the same execution policy types that
          ``RAJA::forall`` loop execution templates do.

Each of the examples below uses the same integer arrays for input
and output values. We set the input array and print them as follows:

.. literalinclude:: ../../../../examples/tut_sort.cpp
   :start-after: _sort_array_init_start
   :end-before: _sort_array_init_end
   :language: C++

This generates the following sequence of values in the ``in`` array::

   6 7 2 1 0 9 4 8 5 3 4 9 6 3 7 0 1 8 2 5

and the following sequence of (key, value) pairs in the ``in`` and ``in_vals``
arrays::

   (6,0) (7,0) (2,0) (1,0) (0,0) (9,0) (4,0) (8,0) (5,0) (3,0)
   (4,1) (9,1) (6,1) (3,1) (7,1) (0,1) (1,1) (8,1) (2,1) (5,1)

^^^^^^^^^^^^^^^^
Unstable Sorts
^^^^^^^^^^^^^^^^

A sequential unstable sort operation is performed by:

.. literalinclude:: ../../../../examples/tut_sort.cpp
   :start-after: _sort_seq_start
   :end-before: _sort_seq_end
   :language: C++

Since no comparator is passed to the sort method, the default less operation
is applied and the result generated in the ``out`` array is non-decreasing sort
on the ``out`` array. The resulting ``out`` array contains the values::

   0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9

We can be explicit about the operation used in the sort by passing the
less operator to the sort method:

.. literalinclude:: ../../../../examples/tut_sort.cpp
   :start-after: _sort_seq_less_start
   :end-before: _sort_seq_less_end
   :language: C++

The result in the ``out`` array is the same.

An unstable parallel sort operation using OpenMP multi-threading is
accomplished similarly by replacing the execution policy type:

.. literalinclude:: ../../../../examples/tut_sort.cpp
   :start-after: _sort_omp_less_start
   :end-before: _sort_omp_less_end
   :language: C++

As is commonly done with RAJA, the only difference between this code and
the previous one is that the execution policy is different. If we want to
run the sort on a GPU using CUDA, we would use a CUDA execution policy. This
will be shown shortly.

^^^^^^^^^^^^^^^^
Stable Sorts
^^^^^^^^^^^^^^^^

A sequential stable sort (less) operation is performed by:

.. literalinclude:: ../../../../examples/tut_sort.cpp
   :start-after: _sort_stable_seq_less_start
   :end-before: _sort_stable_seq_less_end
   :language: C++

This generates the following sequence of values in the output array::

   0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9

Note that the stable sort result is the same as the unstable sort in this case
because we are sorting integers. We will show an example of sorting pairs later
where this is not the case.

Running the same sort operation on a GPU using CUDA is done by:

.. literalinclude:: ../../../../examples/tut_sort.cpp
   :start-after: _sort_stable_cuda_less_start
   :end-before: _sort_stable_cuda_less_end
   :language: C++

Note that we pass the number of threads per CUDA thread block as the template
argument to the CUDA execution policy as we do in other cases.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Other Comparators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using a different comparator allows sorting in a different order.
Here is a sequential stable sort that uses the greater operator:

.. literalinclude:: ../../../../examples/tut_sort.cpp
   :start-after: _sort_stable_seq_greater_start
   :end-before: _sort_stable_seq_greater_end
   :language: C++

This generates the following sequence of values in non-increasing order in
the output array::

   9 9 8 8 7 7 6 6 5 5 4 4 3 3 2 2 1 1 0 0

Note that the only operators provided by RAJA that are valid to use in sort
because they form a strict weak ordering of elements for arithmetic types are
less and greater. Also note that the the cuda sort backend only supports
RAJA's operators less and greater.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Sort Pairs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sort *Pairs* operations generate the same results as the sort operations
we have just described. However, an additional array of values is also permuted
to match the sorted array so **two arrays are passed to sort pairs methods.**

Here is a sequential unstable sort pairs that uses the less operator:

.. literalinclude:: ../../../../examples/tut_sort.cpp
   :start-after: _sort_pairs_seq_less_start
   :end-before: _sort_pairs_seq_less_end
   :language: C++

This generates the following sequence in the output array::

   (0,0) (0,1) (1,0) (1,1) (2,0) (2,1) (3,0) (3,1) (4,0) (4,1)
   (5,1) (5,0) (6,1) (6,0) (7,0) (7,1) (8,0) (8,1) (9,1) (9,0)

Note that some of the pairs with equivalent keys stayed in the same order
they appeared in the unsorted arrays like ``(8,0) (8,1)``, while others are
reversed like ``(9,1) (9,0)``.

Here is a sequential stable sort pairs that uses the greater operator:

.. literalinclude:: ../../../../examples/tut_sort.cpp
   :start-after: _sort_stable_pairs_seq_greater_start
   :end-before: _sort_stable_pairs_seq_greater_end
   :language: C++

This generates the following sequence in the output array::

   (9,0) (9,1) (8,0) (8,1) (7,0) (7,1) (6,0) (6,1) (5,0) (5,1)
   (4,0) (4,1) (3,0) (3,1) (2,0) (2,1) (1,0) (1,1) (0,0) (0,1)

Note that all pairs with equivalent keys stayed in the same order that they
appeared in the unsorted arrays.

As you may expect at this point, running an stable sort pairs
operation using OpenMP is accomplished by:

.. literalinclude:: ../../../../examples/tut_sort.cpp
   :start-after: _sort_stable_pairs_omp_greater_start
   :end-before: _sort_stable_pairs_omp_greater_end
   :language: C++

This generates the following sequence in the output array (as we saw earlier)::

   (9,0) (9,1) (8,0) (8,1) (7,0) (7,1) (6,0) (6,1) (5,0) (5,1)
   (4,0) (4,1) (3,0) (3,1) (2,0) (2,1) (1,0) (1,1) (0,0) (0,1)

and the only difference is the execution policy template parameter.

Lastly, we show a parallel unstable sort pairs operation using CUDA:

.. literalinclude:: ../../../../examples/tut_sort.cpp
   :start-after: _sort_pairs_cuda_greater_start
   :end-before: _sort_pairs_cuda_greater_end
   :language: C++

.. note:: RAJA sorts for the HIP back-end are similar to those for CUDA.

The file ``RAJA/examples/tut_sort.cpp`` contains the complete
working example code.
