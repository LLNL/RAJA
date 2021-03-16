.. ##
.. ## Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/COPYRIGHT file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _atomichist-label:

--------------------------------------------------
Computing a Histogram with Atomic Operations
--------------------------------------------------

Key RAJA features shown in this example:

  * ``RAJA::forall`` loop execution template 
  * ``RAJA::RangeSegment`` iteration space construct
  * RAJA atomic add operation

The example uses an integer array of length 'N' randomly initialized with 
values in the interval [0, M). While iterating over the array, the kernel 
accumulates the number of occurrences of each value in the array using atomic 
add operations. Atomic operations allow one to update a memory location 
referenced by a specific address in parallel without data races. The example 
shows how to use RAJA portable atomic operations and that they are used 
similarly for different programming model back-ends. 

.. note:: Each RAJA reduction operation requires an atomic policy type
          parameter that must be compatible with the execution policy for 
          the kernel in which it is used.

For a complete description of supported RAJA atomic operations and 
atomic policies, please see :ref:`atomics-label`.

All code snippets described below use the loop range:

.. literalinclude:: ../../../../examples/tut_atomic-histogram.cpp
   :start-after: _range_atomic_histogram_start
   :end-before: _range_atomic_histogram_end
   :language: C++

and the integer array 'bins' of length 'M' to accumulate the number of 
occurrences of each value in the array.

Here is the OpenMP version:

.. literalinclude:: ../../../../examples/tut_atomic-histogram.cpp
   :start-after: _rajaomp_atomic_histogram_start
   :end-before: _rajaomp_atomic_histogram_end
   :language: C++

Each slot in the 'bins' array is incremented by one when a value associated 
with that slot is encountered. Note that the ``RAJA::atomicAdd`` 
operation uses an OpenMP atomic policy, which is compatible with the OpenMP 
loop execution policy.

The CUDA and HIP versions are similar:

.. literalinclude:: ../../../../examples/tut_atomic-histogram.cpp
   :start-after: _rajacuda_atomic_histogram_start
   :end-before: _rajacuda_atomic_histogram_end
   :language: C++

and:

.. literalinclude:: ../../../../examples/tut_atomic-histogram.cpp
   :start-after: _rajahip_atomic_histogram_start
   :end-before: _rajahip_atomic_histogram_end
   :language: C++

Here, the atomic add operations uses CUDA and HIP atomic policies, which are 
compatible with the CUDA and HIP loop execution policies.

Note that RAJA provides an ``auto_atomic`` policy for easier usage and 
improved portability. This policy will do the right thing in most 
circumstances. If OpenMP is enabled, the OpenMP atomic policy will be used, 
which is correct in a sequential execution context as well. Otherwise, the 
sequential atomic policy will be applied. Similarly, if it is encountered in 
a CUDA or HIP execution context, the corresponding GPU back-end atomic policy 
will be applied. 

For example, here is the CUDA version that uses the 'auto' atomic policy:

.. literalinclude:: ../../../../examples/tut_atomic-histogram.cpp
   :start-after: _rajacuda_atomicauto_histogram_start
   :end-before: _rajacuda_atomicauto_histogram_end
   :language: C++

and the HIP version:

.. literalinclude:: ../../../../examples/tut_atomic-histogram.cpp
   :start-after: _rajahip_atomicauto_histogram_start
   :end-before: _rajahip_atomicauto_histogram_end
   :language: C++

The same CUDA and HIP loop execution policies as in the previous examples 
are used.

The file ``RAJA/examples/tut_atomic-histogram.cpp`` contains the complete 
working example code.
