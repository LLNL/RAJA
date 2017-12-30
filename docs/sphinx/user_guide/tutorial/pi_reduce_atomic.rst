.. ##
.. ## Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
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
Computing Pi with a Reduction or Atomic Operation
--------------------------------------------------

Key RAJA features shown in this example:

  * ``RAJA::forall`` loop iteration template 
  * ``RAJA::RangeSegment`` iteration space construct
  * RAJA sum reduction type
  * RAJA atomic add operation

The example computes an approximation to pi. It illustrates usage of RAJA 
portable reduction types and atomic operations and shows that they are
used similarly for different programming model back-ends. For a complete 
description of supported RAJA atomic operations, please see 
:ref:`atomics-label`.

For all variants that approximate pi, we define a number of 'bins' and an
index range segment to iterate over the bins. We also allocate a double array 
of length 1, in CUDA Unified Memory if CUDA is enabled, that the atomic
operation 

.. literalinclude:: ../../../../examples/ex8-pi-reduce_vs_atomic.cpp
                    :lines: 56-59

The sequential version that approximates pi using a ``RAJA::ReduceSum`` type
is:

.. literalinclude:: ../../../../examples/ex8-pi-reduce_vs_atomic.cpp
                    :lines: 69-80

Note that to print the value of pi, we retrieve the reduced value using the
'get()' method on the reduction object.

The sequential version that uses an atomic operation is:

.. literalinclude:: ../../../../examples/ex8-pi-reduce_vs_atomic.cpp
                    :lines: 85-95

The parallel variants that use OpenMP multi-threading are similar except
for the execution and reduction policies and the atomic policy. 
Specifically, for the OpenMP reduction variant, the types are:

.. literalinclude:: ../../../../examples/ex8-pi-reduce_vs_atomic.cpp
                    :lines: 104-107

For the atomic variant, the atomic policy type is:

.. literalinclude:: ../../../../examples/ex8-pi-reduce_vs_atomic.cpp
                    :lines: 122-122

For completeness, the CUDA variants employ the types:

.. literalinclude:: ../../../../examples/ex8-pi-reduce_vs_atomic.cpp
                    :lines: 141-144

for the reduction version and:

.. literalinclude:: ../../../../examples/ex8-pi-reduce_vs_atomic.cpp
                    :lines: 159-159

for the atomic variant.

It it worth noting that RAJA provides an ``auto_atomic`` policy for easier
usage and improved portability. This policy should do the right thing in
most circumstances. Specifically, if it is encountered in a CUDA execution
context, the CUDA atomic policy will be applied. If OpenMP is enabled, the 
OpenMP atomic policy will be used, which should be correct in a sequential
execution context. Otherwise, the sequential atomic policy will be applied.
For information about the full range of RAJA atomic support, see
:ref:`atomics-label`.

The file ``RAJA/examples/ex8-pi-reduce_vs_atomic.cpp`` contains the complete 
working example code.
