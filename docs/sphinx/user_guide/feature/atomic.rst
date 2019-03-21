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

.. _atomics-label:

========
Atomics
========

RAJA provides portable atomic operations that can be used to update values
at arbitrary memory locations while avoiding data races. They are described
in this section.

A complete working example code that shows RAJA atomic usage can be found in 
:ref:`atomichist-label`.

.. note:: * All RAJA atomic operations are in the namespace ``RAJA::atomic``.

-----------------
Atomic Operations
-----------------

RAJA atomic support includes a variety of the most common atomic operations.

.. note:: * Each RAJA atomic operation is templated on an *atomic policy*.
          * Each method described in the table below returns the value of 
            the potentially modified argument (i.e., \*acc) immediately before 
            the atomic operation is applied, in case it is needed by a user.
          * Please set the nvcc option "-arch=sm_35" (or greater), when CUDA is enabled. If a CUDA architecture does not support an sm_xy atomic operation, RAJA will generically implement it with atomicCAS.

^^^^^^^^^^^
Arithmetic
^^^^^^^^^^^

* ``atomicAdd< atomic_policy >(T* acc, T value)`` - Add value to \*acc.

* ``atomicSub< atomic_policy >(T* acc, T value)`` - Subtract value from \*acc.

^^^^^^^^^^^
Min/max
^^^^^^^^^^^

* ``atomicMin< atomic_policy >(T* acc, T value)`` - Set \*acc to min of \*acc and value.

* ``atomicMax< atomic_policy >(T* acc, T value)`` - Set \*acc to max of \*acc and value.

^^^^^^^^^^^^^^^^^^^^
Increment/decrement
^^^^^^^^^^^^^^^^^^^^

* ``atomicInc< atomic_policy >(T* acc)`` - Add 1 to \*acc.

* ``atomicDec< atomic_policy >(T* acc)`` - Subtract 1 from \*acc.

* ``atomicInc< atomic_policy >(T* acc, T compare)`` - Add 1 to \*acc if \*acc < compare, else set \*acc to zero.

* ``atomicDec< atomic_policy >(T* acc, T compare)`` - Subtract 1 from \*acc if \*acc != 0 and \*acc <= compare, else set \*acc to compare.

^^^^^^^^^^^^^^^^^^^^
Bitwise operations
^^^^^^^^^^^^^^^^^^^^

* ``atomicAnd< atomic_policy >(T* acc, T value)`` - Bitwise 'and' equivalent: Set \*acc to \*acc & value. Only works with integral data types.

* ``atomicOr< atomic_policy >(T* acc, T value)`` - Bitwise 'or' equivalent: Set \*acc to \*acc | value. Only works with integral data types.

* ``atomicXor< atomic_policy >(T* acc, T value)`` - Bitwise 'xor' equivalent: Set \*acc to \*acc ^ value. Only works with integral data types.

^^^^^^^^^^^^^^^^^^^^
Replace
^^^^^^^^^^^^^^^^^^^^

* ``atomicExchange< atomic_policy >(T* acc, T value)`` - Replace \*acc with value.

* ``atomicCAS< atomic_policy >(T* acc, Tcompare, T value)`` - Compare and swap: Replace \*acc with value if and only if \*acc is equal to compare.

Here is a simple example that shows how to use an atomic operation to compute
an integral sum on a CUDA GPU device::

  //
  // Use CUDA UM to share data pointer with host and device code.
  // RAJA mechanics work the same way if device data allocation
  // and host-device copies are done with traditional cudaMalloc
  // and cudaMemcpy.
  //
  int* sum = nullptr;
  cudaMallocManaged((void **)&sum, sizeof(int));
  cudaDeviceSynchronize();
  *sum = 0;

  RAJA::forall< RAJA::cuda_exec >(RAJA::RangeSegment(0, N), 
    [=] RAJA_DEVICE (RAJA::Index_type i) {

    RAJA::atomic::atomicAdd< RAJA::cuda_atomic >(sum, 1);

  });

After this kernel executes, '*sum' will be equal to 'N'.

^^^^^^^^^^^^^^^^^^^^
AtomicRef
^^^^^^^^^^^^^^^^^^^^

RAJA also provides an atomic interface similar to the C++20 'std::atomic_ref', 
but which works for arbitrary memory locations. The class 
``RAJA::atomic::AtomicRef`` provides an object-oriented interface to the 
atomic methods described above. For example, after the following operations:: 

  double val = 2.0;
  RAJA::atomic::AtomicRef<double,  RAJA::omp_atomic > sum(&val);

  sum++;
  ++sum;
  sum += 1.0; 

the value of 'val' will be 5.

-----------------
Atomic Policies
-----------------

For more information about available RAJA atomic policies, please see
:ref:`atomicpolicy-label`.
