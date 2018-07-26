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

RAJA atomic support includes a range of the most common atomic operations.

.. note:: * Each RAJA atomic operation is templated on an *atomic policy*.
          * Each of methods below returns the value of the potentially modified
            argument (i.e., \*acc) immediately before the atomic operation is 
            applied.

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
Bitwise atomics
^^^^^^^^^^^^^^^^^^^^

* ``atomicAnd< atomic_policy >(T* acc, T value)`` - Bitwise 'and' equivalent: Set \*acc to \*acc & value. Only works with integral data types.

* ``atomicOr< atomic_policy >(T* acc, T value)`` - Bitwise 'or' equivalent: Set \*acc to \*acc | value. Only works with integral data types.

* ``atomicXor< atomic_policy >(T* acc, T value)`` - Bitwise 'xor' equivalent: Set \*acc to \*acc ^ value. Only works with integral data types.

^^^^^^^^^^^^^^^^^^^^
Replace
^^^^^^^^^^^^^^^^^^^^

* ``atomicExchange< atomic_policy >(T* acc, T value)`` - Replace \*acc with value.

* ``atomicCAS< atomic_policy >(T* acc, Tcompare, T value)`` - Compare and swap: Replace \*acc with value if and only if \*acc is equal to compare.

Here is a simple example that shows how to use an atomic method to accumulate
a integral sum on a CUDA GPU device::

  //
  // Use CUDA UM to share data pointer with host and device code.
  // RAJA mechanics works the same way if device data allocation
  // and host-device copies are done with traditional cudaMalloc
  // and cudaMemcpy.
  //
  int* sum = nullptr;
  cudaMallocManaged((void **)&sum, sizeof(int));
  cudaDeviceSynchronize();
  sum = 0;

  RAJA::forall< RAJA::cuda_exec >(RAJA::RangeSegment(0, N), 
    [=] RAJA_DEVICE (RAJA::Index_type i) {

    RAJA::atomic::atomicAdd< RAJA::cuda_atomic >(&sum, 1);

  });

After this operation, 'sum' will be equal to 'N'.

^^^^^^^^^^^^^^^^^^^^
AtomicRef
^^^^^^^^^^^^^^^^^^^^

RAJA also provides an atomic interface similar to C++ 'std::atomic', but which
works for arbitrary memory locations. The class ``RAJA::atomic::AtomicRef`` 
provides an object-oriented interface to the atomic methods described above. 
For example, after the following operations:: 

  double val = 2.0;
  RAJA::atomic::AtomicRef<double,  RAJA::omp_atomic > sum(&val);

  sum++;
  ++sum;
  sum += 1.0; 

the value of 'val' will be 5.

Note that the operations provided by the 'AtomicRef' class return the object
that holds the address of the data given to the constructor. It will likely 
change with each atomic update call. If you need to keep the original value 
of the data before an atomic call, you need to use the atomic methods described 
earlier and not the ``RAJA::atomic::AtomicRef`` interface.

---------------
Atomic Policies
---------------

.. note:: * All RAJA atomic policies are in the namespace ``RAJA::atomic``.
          * There are no RAJA atomic policies for TBB (Intel Threading Building 
            Blocks) execution contexts currently.

* ``seq_atomic``     - Policy for use in sequential execution contexts, primarily for consistency with parallel policies. Note that sequential atomic operations are not protected and will likely produce incorrect results when used in a parallel execution context.

* ``omp_atomic``     - Policy to use 'omp atomic' pragma when applicable; otherwise, revert to builtin compiler atomics.

* ``cuda_atomic``    - Policy to use CUDA atomic operations in GPU device code.

* ``builtin_atomic`` - Policy to use compiler "builtin" atomic operations.

* ``auto_atomic``    - Policy that will attempt to do the "correct thing". For example, in a CUDA execution context, this is equivalent to using the RAJA::cuda_atomic policy; if OpenMP is enabled, the RAJA::omp_atomic policy will be used; otherwise, RAJA::seq_atomic will be applied.

For example, we could use the 'auto_atomic' policy in the example above:: 

  RAJA::forall< RAJA::cuda_exec >(RAJA::RangeSegment seg(0, N), 
    [=] RAJA_DEVICE (RAJA::Index_type i) {

    RAJA::atomic::atomicAdd< RAJA::auto_atomic >(&sum, 1);

  });

Here, the atomic operation knows that it is used within a CUDA execution 
context and does the right thing. Similarly, if the 'forall' method used 
an OpenMP execution policy, the OpenMP version of the atomic operation 
would be used.
