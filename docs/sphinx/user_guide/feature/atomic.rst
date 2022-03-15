.. ##
.. ## Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
.. ## and other RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
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

.. note:: * All RAJA atomic operations are in the namespace ``RAJA``.

.. _atomic-ops:

-----------------
Atomic Operations
-----------------

RAJA atomic support includes a variety of the most common atomic operations.

.. note:: * Each RAJA atomic operation is templated on an *atomic policy*.
          * Each method described in the table below returns the value of 
            the potentially modified argument (i.e., \*acc) immediately before 
            the atomic operation is applied, in case it is needed by a user.
          * See :ref:`atomics-label` for details about CUDA atomic operations.

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

  RAJA::forall< RAJA::cuda_exec<BLOCK_SIZE> >(RAJA::RangeSegment(0, N), 
    [=] RAJA_DEVICE (RAJA::Index_type i) {

    RAJA::atomicAdd< RAJA::cuda_atomic >(sum, 1);

  });

After this kernel executes, the value reference by 'sum' will be 'N'.

^^^^^^^^^^^^^^^^^^^^
AtomicRef
^^^^^^^^^^^^^^^^^^^^

RAJA also provides an atomic interface similar to the C++20 'std::atomic_ref', 
but which works for arbitrary memory locations. The class 
``RAJA::AtomicRef`` provides an object-oriented interface to the 
atomic methods described above. For example, after the following operations:: 

  double val = 2.0;
  RAJA::AtomicRef<double,  RAJA::omp_atomic > sum(&val);

  sum++;
  ++sum;
  sum += 1.0; 

the value of 'val' will be 5.

-----------------
Atomic Policies
-----------------

For more information about available RAJA atomic policies, please see
:ref:`atomicpolicy-label`.


.. _cudaatomics-label:

---------------------------------------
CUDA Atomics Architecture Dependencies
---------------------------------------

The internal implementations for RAJA atomic operations may vary depending
on which CUDA architecture is available and/or specified when the RAJA
is configured for compilation. The following rules apply when the following
CUDA architecture level is chosen:

  * **CUDA architecture is lower than `sm_35`** 

    * Certain atomics will be implemented using CUDA `atomicCAS` 
      (Compare and Swap).

  * **CUDA architecture is `sm_35` or higher**   

    * CUDA native 64-bit unsigned atomicMin, atomicMax, atomicAnd, atomicOr,
      atomicXor are used.

  * **CUDA architecture is `sm_60` or higher** 

    * CUDA native 64-bit double `atomicAdd` is used.

---------------------
DESUL Atomics Support
---------------------

RAJA provides support for the use of `DESUL Atomics <https://github.com/desul/desul>`_ as
an alternative backend to the default implementation of RAJA atomics. DESUL atomics are considered an **experimental** feature in RAJA at this point. DESUL atomics
may impact the performance of some atomic functions. While switching
to DESUL atomics typically yields positive or neutral performance results, some atomic
operations may perform worse when using DESUL.

To enable DESUL Atomics:

#. Ensure that RAJA and its dependencies are configured to use C++14.
#. Set ``RAJA_ENABLE_DESUL_ATOMICS=On``.

Enabling DESUL Atomics alters RAJA atomic functions to be wrapper-functions for their
DESUL counterparts. This removes the need for user code changes to switch between
DESUL and RAJA implementations. The exception to this is when RAJA atomic helper functions
are used instead of the backwards-compatible API functions specified by :ref:`atomic-ops`. By *helper functions*, we mean the RAJA atomic methods which take a reduction policy object as the first argument, instead of specifying the reduction policy type as a template parameter. 

DESUL atomic functions are compiled with the proper back-end implementation based on the scope in which they are
called, which removes the need to specify atomic policies for
target back-ends. As a result, atomic policies such as ``cuda_atomic`` or ``omp_atomic``
are ignored when DESUL is enabled, but are still necessary to pass in as parameters
to the RAJA API. This will likely change in the future and RAJA atomic policies will be removed.
