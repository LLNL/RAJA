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

.. _atomics-label:

========
Atomics
========

To avoid race conditions at specific memory locations, RAJA provides 
portable atomic operation, each of which is templated on an *atomic policy*. 
This section describes the atomic operations and policies available in RAJA.

-----------------
Atomic Operations
-----------------

.. note:: All RAJA atomic operations are in the namespace ``RAJA::atomic``.

* ``atomicAdd<Policy>(T* acc, T value)`` - Add value to \*acc.

* ``atomicSub<AtomicPolicy>(T* acc, T value)`` - Subtract value from \*acc.

* ``atomicMin<AtomicPolicy>(T* acc, T value)`` - Set \*acc to min of \*acc and value.

* ``atomicMax<AtomicPolicy>(T* acc, T value)`` - Set \*acc to max of \*acc and value.

* ``atomicInc<AtomicPolicy>(T* acc)`` - Add 1 to \*acc.

* ``atomicDec<AtomicPolicy>(T* acc)`` - Subtract 1 from \*acc.

* ``atomicInc<AtomicPolicy>(T* acc, T compare)`` - Add 1 to \*acc if \*acc < compare, else set \*acc to zero.

* ``atomicDec<AtomicPolicy>(T* acc, T compare)`` - Subtract 1 from \*acc if \*acc != 0 and \*acc <= compare, else set \*acc to compare.

* ``atomicDec<AtomicPolicy>(T* acc, T compare)`` - Subtract 1 from \*acc if \*acc != 0 and \*acc <= compare, else set \*acc to compare.

* ``atomicAnd<AtomicPolicy>(T* acc, T value)`` - Bitwise 'and' equivalent: Set \*acc to \*acc & value. Only works with integral data types.

* ``atomicOr<AtomicPolicy>(T* acc, T value)`` - Bitwise 'or' equivalent: Set \*acc to \*acc | value. Only works with integral data types.

* ``atomicXor<AtomicPolicy>(T* acc, T value)`` - Bitwise 'xor' equivalent: Set \*acc to \*acc ^ value. Only works with integral data types.

* ``atomicExchange<AtomicPolicy>(T* acc, T value)`` - Replace \*acc with value.

* ``atomicCAS<AtomicPolicy>(T* acc, Tcompare, T value)`` - Compare and swap: Replace \*acc with value if and only if \*acc is equal to compare.

.. note:: Each of these methods returns the value of \*acc before the atomic
          operation is applied.

RAJA also provides an atomic interface similar to 'std::atomic', but for 
arbitrary memory locations. The class ``RAJA::atomic::AtomicRef`` provides
an object-oriented interface to the atomic methods described above. For 
example, after the following operations:: 

  double val = 2.0;
  RAJA::atomic::AtomicRef<double, AtomicPolicy> sum(&val);

  sum++;
  ++sum;
  sum += 1.0; 

the value of 'val' will be 5.

However, the operators that the 'AtomicRef' class provide return the object
which holds the address of the data given at construction. If you need to keep 
the original value of the data before the atomic call, you need to use the
atomic methods listed above.

---------------
Atomic Policies
---------------

.. note:: All RAJA atomic policies are in the namespace ``RAJA::atomic``.

* ``seq_atomic``     - Policy for use in sequential execution contexts, primarily for consistency with parallel policies. Note that sequential atomic operations are not protected and will likely produce incorrect results when used in a parallel execution context.

* ``auto_atomic``    - Policy that will attempt to do the "correct thing". For example, in a CUDA execution context, this is equivalent to using the RAJA::cuda_atomic policy; if OpenMP is enabled, the RAJA::omp_atomic policy will be used; otherwise, RAJA::seq_atomic will be applied.

* ``buildin_atomic`` - Policy to use compiler "builtin" atomic operations.

* ``omp_atomic``     - Policy to use 'omp atomic' pragma when applicable; otherwise, revert to builtin compiler atomics.

* ``cuda_atomic``    - Policy to use CUDA atomic operations in GPU device code.

.. note:: There are no RAJA atomic policies for TBB (Intel Threading Building 
          Blocks) execution contexts.

An simple atomic usage example can be found in ``RAJA/examples/example-atomic-pi.cpp``. 
