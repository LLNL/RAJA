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
portable atomic operations, which are described in this section.

-----------------
Atomic Operations
-----------------

.. note:: * All RAJA atomic operations are in the namespace ``RAJA::atomic``.
          * Each RAJA atomic operation is templated on an *atomic policy*.

* ``atomicAdd< atomic_policy >(T* acc, T value)`` - Add value to \*acc.

* ``atomicSub< atomic_policy >(T* acc, T value)`` - Subtract value from \*acc.

* ``atomicMin< atomic_policy >(T* acc, T value)`` - Set \*acc to min of \*acc and value.

* ``atomicMax< atomic_policy >(T* acc, T value)`` - Set \*acc to max of \*acc and value.

* ``atomicInc< atomic_policy >(T* acc)`` - Add 1 to \*acc.

* ``atomicDec< atomic_policy >(T* acc)`` - Subtract 1 from \*acc.

* ``atomicInc< atomic_policy >(T* acc, T compare)`` - Add 1 to \*acc if \*acc < compare, else set \*acc to zero.

* ``atomicDec< atomic_policy >(T* acc, T compare)`` - Subtract 1 from \*acc if \*acc != 0 and \*acc <= compare, else set \*acc to compare.

* ``atomicDec< atomic_policy >(T* acc, T compare)`` - Subtract 1 from \*acc if \*acc != 0 and \*acc <= compare, else set \*acc to compare.

* ``atomicAnd< atomic_policy >(T* acc, T value)`` - Bitwise 'and' equivalent: Set \*acc to \*acc & value. Only works with integral data types.

* ``atomicOr< atomic_policy >(T* acc, T value)`` - Bitwise 'or' equivalent: Set \*acc to \*acc | value. Only works with integral data types.

* ``atomicXor< atomic_policy >(T* acc, T value)`` - Bitwise 'xor' equivalent: Set \*acc to \*acc ^ value. Only works with integral data types.

* ``atomicExchange< atomic_policy >(T* acc, T value)`` - Replace \*acc with value.

* ``atomicCAS< atomic_policy >(T* acc, Tcompare, T value)`` - Compare and swap: Replace \*acc with value if and only if \*acc is equal to compare.

.. note:: Each of these methods returns the value of \*acc before the atomic
          operation is applied.

Here is a simple example that shows how to use an atomic method to accumulate
a integral sum on a CUDA GPU device::

  RAJA::RangeSegment seg(0, N);

  int sum = 0;

  RAJA::forall<RAJA::cuda_exec>(seg, [=] RAJA_DEVICE (RAJA::Index_type i) {
    RAJA::atomic::atomicAdd<cuda_atomic>(&sum, 1);
  }

After this operation, 'sum' will be equal to 'N'.

RAJA also provides an atomic interface similar to C++ 'std::atomic', but for 
arbitrary memory locations. The class ``RAJA::atomic::AtomicRef`` provides
an object-oriented interface to the atomic methods described above. For 
example, after the following operations:: 

  double val = 2.0;
  RAJA::atomic::AtomicRef<double,  atomic_policy > sum(&val);

  sum++;
  ++sum;
  sum += 1.0; 

the value of 'val' will be 5.

Note that the operators that the 'AtomicRef' class provide return the object
that holds the address of the data given passes the constructor. If you need 
to keep the original value of the data before the atomic call, you need to 
use the atomic methods listed above.

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

For example, we could use the 'auto_atomic' policy in the example above:: 

  RAJA::RangeSegment seg(0, N);

  int sum = 0;

  RAJA::forall<RAJA::cuda_exec>(seg, [=] RAJA_DEVICE (RAJA::Index_type i) {
    RAJA::atomic::atomicAdd<auto_atomic>(&sum, 1);
  }

Here, the atomic operation knows that it is used within a CUDA execution 
context and does the right thing. Similarly, if the 'forall' method used 
an OpenMP execution policy, the OpenMP version of the atomic operation 
would be used.

Another simple example of atomic usage can be found in 
``RAJA/examples/example-atomic-pi.cpp``. 
