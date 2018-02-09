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

.. _pi-label:

--------------------------------------------------
Computing a Histogram with Atomic Operations
--------------------------------------------------

Key RAJA features shown in this example:

  * ``RAJA::forall`` loop iteration template 
  * ``RAJA::RangeSegment`` iteration space construct
  * RAJA atomic add operation

The example uses an integer array of length 'N' randomly initialized with 
values in the interval [0, M). It iterates over the array and accumulates the 
number of occurrences of each value in the array using atomic add operations. 
It illustrates usage of RAJA portable atomic operations and shows that they are
used similarly for different programming model back-ends. For a complete 
description of supported RAJA atomic operations, please see 
:ref:`atomics-label`.

All code snippets described below use the loop range:

.. literalinclude:: ../../../../examples/ex10-binning.cpp
                    :lines: 57-57

and the integer array 'bins' of length 'M' to accumulate the number of 
occurrences of each value in the array.

Here is the OpenMP version:

.. literalinclude:: ../../../../examples/ex10-binning.cpp
                    :lines: 90-97

Each slot in the 'bins' array is incremented by one when a value associated 
with that slot is encountered. Note that the ``RAJA::atomic::atomicAdd`` 
operation uses an OpenMP atomic policy, which is compatible with the OpenMP 
loop execution policy.

The CUDA version is similar:

.. literalinclude:: ../../../../examples/ex10-binning.cpp
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

.. literalinclude:: ../../../../examples/ex10-binning.cpp
                    :lines: 142-148

The same CUDA loop execution policy as in the previous example is used.

For information about the full range of RAJA atomic support, see
:ref:`atomics-label`.

The file ``RAJA/examples/ex10-binning.cpp`` contains the complete 
working example code.
