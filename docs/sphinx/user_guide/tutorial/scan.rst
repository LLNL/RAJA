.. ##
.. ## Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _scan-label:

--------------------------------------------------
Parallel Scan Operations
--------------------------------------------------

This section contains a working exercise in the file 
``RAJA/exercises/scan.cpp``. 

Key RAJA features shown in this section are:

  * ``RAJA::inclusive_scan`` operation
  * ``RAJA::inclusive_scan_inplace`` operation
  * ``RAJA::exclusive_scan`` operation
  * ``RAJA::exclusive_scan_inplace`` operation
  * RAJA operators for different types of scans; e.g., plus, minimum, maximum, etc.

The file ``RAJA/exercises/scan_solution.cpp`` contains working code for the
examples discussed in this section.

In this section, we present examples of RAJA sequential, OpenMP,
and CUDA scan operations and show how different scan operations can be 
performed by passing different RAJA operators to the RAJA scan template 
methods. Each operator is a template type, where the template argument is 
the type of the values it operates on. For a summary of RAJA scan 
functionality, please see :ref:`scan-label`. 

.. note:: RAJA scan operations use the same execution policy types that 
          ``RAJA::forall`` loop execution templates do.

.. note:: RAJA scan operations take 'span' arguments to express the sequential
          index range of array entries used in the scan. Typically, these
          scan objects are created using the ``RAJA::make_span`` method
          as shown in the examples below.

Each of the examples below uses the same integer arrays for input
and output values. We set the input array and print them as such:

.. literalinclude:: ../../../../exercises/scan_solution.cpp
   :start-after: _scan_array_init_start
   :end-before: _scan_array_init_end
   :language: C++

This generates the following sequence of values. This sequence will be used as 
the 'in' array for each of the following examples.::

   -1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18

^^^^^^^^^^^^^^^^
Inclusive Scans
^^^^^^^^^^^^^^^^

RAJA's scan operations are intended to be used as standalone operations, and do not 
need to be used within RAJA execution context such as ``RAJA::forall``. A sequential
inclusive scan operation can be executed like so:

.. literalinclude:: ../../../../exercises/scan_solution.cpp
   :start-after: _scan_inclusive_seq_start
   :end-before: _scan_inclusive_seq_end
   :language: C++

Since no operator is passed to the scan method, the default 'sum' operation 
is applied and the result generated in the 'out' array is a prefix-sum based 
on the 'in' array. The resulting 'out' array contains the values::

   -1 -1 0 2 5 9 14 20 27 35 44 54 65 77 90 104 119 135 152 170

We can be explicit about the operation used in the scan by passing the RAJA
'plus' operator ``RAJA::operators::plus<int>`` to the scan method:

.. literalinclude:: ../../../../exercises/scan_solution.cpp
   :start-after: _scan_inclusive_seq_plus_start
   :end-before: _scan_inclusive_seq_plus_end
   :language: C++

The result in the 'out' array is the same as above::

   -1 -1 0 2 5 9 14 20 27 35 44 54 65 77 90 104 119 135 152 170

An inclusive parallel scan operation using OpenMP multithreading is
accomplished similarly by replacing the execution policy type:

.. literalinclude:: ../../../../exercises/scan_solution.cpp
   :start-after: _scan_inclusive_omp_plus_start
   :end-before: _scan_inclusive_omp_plus_end
   :language: C++

As expected, this produces the same result as the previous two examples.

As is commonly the case with RAJA, the only difference between this code and
the previous one is the execution policy. If we want to 
run the scan on a GPU using CUDA, we would use a CUDA execution policy as
is shown in examples below.

^^^^^^^^^^^^^^^^
Exclusive Scans
^^^^^^^^^^^^^^^^

A sequential exclusive scan (plus) operation is performed by:

.. literalinclude:: ../../../../exercises/scan_solution.cpp
   :start-after: _scan_exclusive_seq_plus_start
   :end-before: _scan_exclusive_seq_plus_end
   :language: C++

This generates the following sequence of values in the output array::

   0 -1 -1 0 2 5 9 14 20 27 35 44 54 65 77 90 104 119 135 152

The result of an exclusive scan is similar to the result of an 
inclusive scan, but differs in two ways. First, the first entry in 
the exclusive scan result is the `identity` of the operator used.
In the example here, it is zero, since the operator is 'plus'. 
Second, the output sequence is shifted one position to the right
when compared to an inclusive scan.

.. note:: The `identity` of an operator is the default value of a given type
          for that operation. For example:
          - The identity of an int for a sum operation is 0.
          - The identity of an int for a maximum operation is -2147483648.


Running the same scan operation on a GPU using CUDA is done by:

.. literalinclude:: ../../../../exercises/scan_solution.cpp
   :start-after: _scan_exclusive_cuda_plus_start
   :end-before: _scan_exclusive_cuda_plus_end
   :language: C++

Note that we pass the number of threads per CUDA thread block as the template
argument to the CUDA execution policy as we do when using ``RAJA::forall``.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In-place Scans and Other Operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*In-place* scan variants generate the same results as the scan operations
we have just described. However, the result is generated in the input array 
directly so **only one array is passed to in-place scan methods.**

Here is a sequential inclusive in-place scan that uses the 'minimum' operator:

.. literalinclude:: ../../../../exercises/scan_solution.cpp
   :start-after: _scan_inclusive_inplace_seq_min_start
   :end-before: _scan_inclusive_inplace_seq_min_end
   :language: C++

Note that, in the example before the scan operation is called, we copy the 
input array into the output array so the result is generated in the output 
array. Doing this, we avoid having to re-initialize the input array to use 
it in other examples. 

This generates the following sequence in the output array::

   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1

Here is a sequential exclusive in-place scan that uses the 'maximum' operator:

.. literalinclude:: ../../../../exercises/scan_solution.cpp
   :start-after: _scan_exclusive_inplace_seq_max_start
   :end-before: _scan_exclusive_inplace_seq_max_end
   :language: C++

This generates the following sequence in the output array::

   -2147483648 -1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17

Since it is an exclusive scan, the first value in the result is the negative 
of the max int value, which is the identity of the maximum operator.

As you may expect at this point, running an exclusive in-place prefix-sum
operation using OpenMP is accomplished by: 

.. literalinclude:: ../../../../exercises/scan_solution.cpp
   :start-after: _scan_exclusive_inplace_omp_plus_start
   :end-before: _scan_exclusive_inplace_omp_plus_end
   :language: C++

This generates the following sequence in the output array (as we saw earlier)::

   0 -1 -1 0 2 5 9 14 20 27 35 44 54 65 77 90 104 119 135 152

and the only difference is the execution policy template parameter.

Lastly, we show a parallel inclusive in-place prefix-sum operation using CUDA:

.. literalinclude:: ../../../../exercises/scan_solution.cpp
   :start-after: _scan_inclusive_inplace_cuda_plus_start
   :end-before: _scan_inclusive_inplace_cuda_plus_end
   :language: C++

and the same for HIP:

.. literalinclude:: ../../../../exercises/scan_solution.cpp
   :start-after: _scan_inclusive_inplace_hip_plus_start
   :end-before: _scan_inclusive_inplace_hip_plus_end
   :language: C++

.. note:: RAJA scans for the HIP back-end are similar to those for CUDA.

