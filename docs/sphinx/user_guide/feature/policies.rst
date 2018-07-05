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

.. _policies-label:

==================
Execution Policies
==================

.. warning:: **This section is a work-in-progress!! It needs to be updated
             and reworked to be consistent with recent changes related to
             new 'kernel' stuff.**

This section describes the various execution policies that ``RAJA`` provides.

.. note:: * All RAJA execution policies are in the namespace ``RAJA``.


-------------------------------
RAJA Forall and Kernel Policies
-------------------------------
The following list of policies may be used with either ``RAJA::forall`` or ``RAJA::kernel`` methods.


**Serial/SIMD Policies**


* ``seq_exec``  - Strictly sequential loop execution.
* ``simd_exec`` - Forced SIMD execution by adding vectorization hints.
* ``loop_exec`` - Allows the compiler to generate whichever optimizations (e.g., SIMD) that it thinks are appropriate.

**OpenMP Policies**

* ``omp_parallel_for_exec`` - Create a parallel region and distributes loop iterations across threads.
* ``omp_for_exec`` - Distribute loop iterations across threads within a parallel region.
* ``omp_for_static`` - Distribute loop iterations across threads using a static schedule within a parallel region.
* ``omp_for_nowait_exec`` - Execute loop in a parallel region and removes synchronization via `nowait` clause.

**Intel Threading Building Blocks (TBB) Policies**

* ``tbb_for_exec`` - Schedule tasks to operate in parallel.
* ``tbb_for_static`` - Implement the parallel_for method using a static scheduler.
* ``tbb_for_dynamic`` - Implement the parallel_for method and uses a dynamic scheduler.

-------------------------------
RAJA Forall Policies
-------------------------------
The following list of policies may only be used with the ``RAJA::forall`` method.

**Serial Policies**
* ``seq_segit`` - Iterate over an index set segment sequentially.

**OpenMP Policies**

* ``omp_parallel_segit`` - Iterate over an index set segments in parallel.
* ``omp_parallel_for_segit`` - Same as above.

**Intel Threading Building Blocks (TBB) Policies**

* ``tbb_segit`` - Iterate over an index set segments in parallel.

**CUDA Policies**

* ``cuda_exec<STRIDE_SIZE>`` - Map a loop to thread blocks with ``STRIDE_SIZE`` threads.

The ``cuda_exec`` policy defines a default thread block size of 256 threads, if no
argument is provided.

--------------------
RAJA Kernel Policies
--------------------

The following list of policies may only be used with the ``RAJA::kernel`` method.

**CUDA Policies**

* ``cuda_block_exec`` - Map a loop level to CUDA thread blocks.
* ``cuda_thread_exec`` - Map a loop level to block local CUDA threads.
* ``cuda_threadblock_exec<STRIDE_SIZE>`` - Map a loop level to thread blocks with ``STRIDE_SIZE`` threads.

--------------------
RAJA Region Policies
--------------------

The following list of policies may only be used with the ``RAJA::region`` method.

* ``seq_region_exec`` - Creates a sequential region.
* ``omp_parallel_region_exec`` - Create an OpenMP parallel region.
