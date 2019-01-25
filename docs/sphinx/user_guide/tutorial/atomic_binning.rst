.. ##
.. ## Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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

.. literalinclude:: ../../../../examples/tut_atomic-binning.cpp
                    :lines: 57-57

and the integer array 'bins' of length 'M' to accumulate the number of 
occurrences of each value in the array.

Here is the OpenMP version:

.. literalinclude:: ../../../../examples/tut_atomic-binning.cpp
                    :lines: 90-97

Each slot in the 'bins' array is incremented by one when a value associated 
with that slot is encountered. Note that the ``RAJA::atomic::atomicAdd`` 
operation uses an OpenMP atomic policy, which is compatible with the OpenMP 
loop execution policy.

The CUDA version is similar:

.. literalinclude:: ../../../../examples/tut_atomic-binning.cpp
                    :lines: 126-133

Here, the atomic add operation uses a CUDA atomic policy, which is compatible 
with the CUDA loop execution policy.

Note that RAJA provides an ``auto_atomic`` policy for easier usage and 
improved portability. This policy will do the right thing in most 
circumstances. Specifically, if it is encountered in a CUDA execution
context, the CUDA atomic policy will be applied. If OpenMP is enabled, the 
OpenMP atomic policy will be used, which should be correct in a sequential
execution context as well. Otherwise, the sequential atomic policy will be 
applied. 

For example, here is the CUDA version that uses the 'auto' atomic policy:

.. literalinclude:: ../../../../examples/tut_atomic-binning.cpp
                    :lines: 142-148

The same CUDA loop execution policy as in the previous example is used.

The file ``RAJA/examples/tut_atomic-binning.cpp`` contains the complete 
working example code.
