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
          * Each of methods described in the table below returns the value of 
            the potentially modified argument (i.e., \*acc) immediately before 
            the atomic operation is applied, in case it is needed by a user.

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

.. _atomicpolicy-label:

---------------
Atomic Policies
---------------

.. note:: * All RAJA atomic policies are in the namespace ``RAJA::atomic``.
          * There are no RAJA atomic policies for TBB (Intel Threading Building 
            Blocks) execution contexts currently.

* ``seq_atomic``     - Policy for use in sequential execution contexts, such as when using RAJA `seq_exec` or `loop_exec` execution policies. RAJA provides sequential atomic policies for consistency with parallel policies, so that sequential and parallel execution policies may be swapped without altering loop kernel code. Note that sequential atomic operations will likely produce incorrect results when used in a parallel execution context.

* ``omp_atomic``     - Policy to use with OpenMP loop execution policies; i.e., they apply the 'omp atomic' pragma when applicable and revert to builtin compiler atomics otherwise.

* ``cuda_atomic``    - Policy to use CUDA atomic operations in GPU device code; i.e., with CUDA execution polcies.

* ``builtin_atomic`` - Policy to use compiler "builtin" atomic operations.

* ``auto_atomic``    - Policy that will attempt to do the "correct thing" without requiring an atomic policy change when a loop  execution policy is changed. For example, in a CUDA execution context, this is equivalent to using the RAJA::cuda_atomic policy; if OpenMP is enabled, the RAJA::omp_atomic policy will be used; otherwise, RAJA::seq_atomic will be applied.

To illustrate, we could use the 'auto_atomic' policy in the example above:: 

  RAJA::forall< RAJA::cuda_exec >(RAJA::RangeSegment seg(0, N), 
    [=] RAJA_DEVICE (RAJA::Index_type i) {

    RAJA::atomic::atomicAdd< RAJA::auto_atomic >(&sum, 1);

  });

Here, the atomic operation knows that it is used within a CUDA execution 
context and the CUDA atomic operation is applied. Similarly, if the 'forall' 
method used an OpenMP execution policy, the OpenMP version of the atomic 
operation would be used.
