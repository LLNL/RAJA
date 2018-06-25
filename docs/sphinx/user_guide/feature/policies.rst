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

--------------------
Serial/SIMD Policies
--------------------

* ``seq_exec``  - Strictly sequential loop execution.
* ``simd_exec`` - Forced SIMD execution by adding vectorization hints.
* ``loop_exec`` - Allows the compiler to generate whichever optimizations (e.g., SIMD that it thinks are appropriate).

---------------
OpenMP Policies
---------------

* ``omp_parallel_for_exec`` - Create a parallel region and distributes loop iterations across threads.
* ``omp_parallel_exec`` - Create a parallel region.
* ``omp_for_exec`` - Distribute loop iterations across threads within a parallel region.
* ``omp_for_static`` - Distribute loop iterations across threads using a static schedule.
* ``omp_for_nowait_exec`` - Execute loop in parallel region and removes synchronization via `nowait` clause.

* ``omp_parallel_segit`` - Iterate over a index set segments in parallel.
* ``omp_parallel_for_segit`` - Same as above.

----------------------
OpenMP Target Policies
----------------------

* ``omp_target_parallel_for_exec`` - Execute loop body in a device (e.g., GPU) environment. Takes a parameter for number of thread teams.

----------------------------------------------
Intel Threading Building Blocks (TBB) Policies
----------------------------------------------

* ``tbb_for_exec`` - Schedule tasks to operate in parallel.
* ``tbb_for_static`` - Implement the parallel_for method using a static scheduler.
* ``tbb_for_dynamic`` - Implement the parallel_for method and uses a dynamic scheduler.

* ``tbb_segit`` - Iterate over a index set segments in parallel.

-------------
CUDA Policies
-------------

Following the CUDA nomenclature, GPU computations are performed on a
grid of threads. Each unit of the grid is referred to as a thread and
threads can be further grouped into thread blocks. As a starting point,
the following policy may be used with the ``RAJA::forall`` loop

* ``cuda_exec<STRIDE_SIZE>`` - Map a loop to thread blocks with ``STRIDE_SIZE`` threads.

The ``cuda_exec`` policy defines a default thread block size of 256 threads, if no
argument is provided. For better control of mapping blocks and block local threads
to loop levels we recommend using the ``RAJA::kernel`` method. A kernel policy list
can map thread blocks and threads to arbitrary loop levels. Mapping global
threads to a loop level is accomplished using the following execution policy:

* ``cuda_threadblock_exec<STRIDE_SIZE>`` - Map a loop nest to thread blocks with ``STRIDE_SIZE`` threads to a loop level.

Finally, mapping a thread block and block local threads to loop levels is done using the
following execution policies:

* ``cuda_block_exec`` - Map a nested loop level to a CUDA thread blocks.

* ``cuda_thread_exec`` - Map a nested loop level to a block local CUDA threads.
