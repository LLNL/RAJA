.. ##
.. ## Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/COPYRIGHT file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _scan-label:

--------------------------------------------------
Parallel Scan Operations
--------------------------------------------------

Key RAJA features shown in this section:

  * ``RAJA::inclusive_scan`` operation
  * ``RAJA::inclusive_scan_inplace`` operation
  * ``RAJA::exclusive_scan`` operation
  * ``RAJA::exclusive_scan_inplace`` operation
  * RAJA operators for different types of scans; e.g., plus, minimum, maximum, etc.

Below, we present examples of RAJA sequential, OpenMP,
and CUDA scan operations and show how different scan operations can be 
performed by passing different RAJA operators to the RAJA scan template 
methods. Each operator is a template type, where the template argument is 
the type of the values it operates on. For a summary of RAJA scan 
functionality, please see :ref:`scan-label`. 

.. note:: RAJA scan operations use the same execution policy types that 
          ``RAJA::forall`` loop execution templates do.

Each of the examples below uses the same integer arrays for input
and output values. We set the input array and print them as follows:

.. literalinclude:: ../../../../examples/tut_scan.cpp
   :start-after: _scan_array_init_start
   :end-before: _scan_array_init_end
   :language: C++

This generates the following sequence of values in the 'in' array::

   3 -1 2 15 7 5 17 9 6 18 1 10 0 14 13 4 11 12 8 16

^^^^^^^^^^^^^^^^
Inclusive Scans
^^^^^^^^^^^^^^^^

A sequential inclusive scan operation is performed by:

.. literalinclude:: ../../../../examples/tut_scan.cpp
   :start-after: _scan_inclusive_seq_start
   :end-before: _scan_inclusive_seq_end
   :language: C++

Since no operator is passed to the scan method, the default 'sum' operation 
is applied and the result generated in the 'out' array is a prefix-sum based 
on the 'in' array. The resulting 'out' array contains the values::

   3 2 4 19 26 31 48 57 63 81 82 92 92 106 119 123 134 146 154 170

We can be explicit about the operation used in the scan by passing the 
'plus' operator to the scan method:

.. literalinclude:: ../../../../examples/tut_scan.cpp
   :start-after: _scan_inclusive_seq_plus_start
   :end-before: _scan_inclusive_seq_plus_end
   :language: C++

The result in the 'out' array is the same.

An inclusive parallel scan operation using OpenMP multithreading is
accomplished similarly by replacing the execution policy type:

.. literalinclude:: ../../../../examples/tut_scan.cpp
   :start-after: _scan_inclusive_omp_plus_start
   :end-before: _scan_inclusive_omp_plus_end
   :language: C++

As is commonly done with RAJA, the only difference between this code and
the previous one is that the execution policy is different. If we want to 
run the scan on a GPU using CUDA, we would use a CUDA execution policy. This
will be shown shortly.

^^^^^^^^^^^^^^^^
Exclusive Scans
^^^^^^^^^^^^^^^^

A sequential exclusive scan (plus) operation is performed by:

.. literalinclude:: ../../../../examples/tut_scan.cpp
   :start-after: _scan_exclusive_seq_plus_start
   :end-before: _scan_exclusive_seq_plus_end
   :language: C++

This generates the following sequence of values in the output array::

   0 3 2 4 19 26 31 48 57 63 81 82 92 92 106 119 123 134 146 154

Note that the exclusive scan result is different than the inclusive scan 
result in two ways. The first entry in the result is the `identity` of the
operator used (here, it is zero, since the operator is 'plus') and, after
that, the output sequence is shifted one position to the right.

Running the same scan operation on a GPU using CUDA is done by:

.. literalinclude:: ../../../../examples/tut_scan.cpp
   :start-after: _scan_exclusive_cuda_plus_start
   :end-before: _scan_exclusive_cuda_plus_end
   :language: C++

Note that we pass the number of threads per CUDA thread block as the template
argument to the CUDA execution policy as we do in other cases.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In-place Scans and Other Operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*In-place* scan operations generate the same results as the scan operations
we have just described. However, the result is generated in the input array 
directly so **only one array is passed to in-place scan methods.**

Here is a sequential inclusive in-place scan that uses the 'minimum' operator:

.. literalinclude:: ../../../../examples/tut_scan.cpp
   :start-after: _scan_inclusive_inplace_seq_min_start
   :end-before: _scan_inclusive_inplace_seq_min_end
   :language: C++

Note that, before the scan, we copy the input array into the output array so 
the result is generated in the output array. Doing this, we avoid having to
re-initialize the input array to use it in other examples. 

This generates the following sequence in the output array::

   3 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1

Here is a sequential exclusive in-place scan that uses the 'maximum' operator:

.. literalinclude:: ../../../../examples/tut_scan.cpp
   :start-after: _scan_exclusive_inplace_seq_max_start
   :end-before: _scan_exclusive_inplace_seq_max_end
   :language: C++

This generates the following sequence in the output array::

   -2147483648 3 3 3 15 15 15 17 17 17 18 18 18 18 18 18 18 18 18 18

Note that the first value in the result is the negative of the max int value;
i.e., the identity of the maximum operator.

As you may expect at this point, running an exclusive in-place prefix-sum
operation using OpenMP is accomplished by: 

.. literalinclude:: ../../../../examples/tut_scan.cpp
   :start-after: _scan_exclusive_inplace_omp_plus_start
   :end-before: _scan_exclusive_inplace_omp_plus_end
   :language: C++

This generates the following sequence in the output array (as we saw earlier)::

   0 3 2 4 19 26 31 48 57 63 81 82 92 92 106 119 123 134 146 15

and the only difference is the execution policy template parameter.

Lastly, we show a parallel inclusive in-place prefix-sum operation using CUDA:

.. literalinclude:: ../../../../examples/tut_scan.cpp
   :start-after: _scan_inclusive_inplace_cuda_plus_start
   :end-before: _scan_inclusive_inplace_cuda_plus_end
   :language: C++

.. note:: RAJA scans for the HIP back-end are similar to those for CUDA.

The file ``RAJA/examples/tut_scan.cpp`` contains the complete 
working example code.
