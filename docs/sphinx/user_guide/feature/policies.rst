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
threads are furthered grouped into thread blocks. Threads and thread blocks 
may have one, two, or three-dimensional indexing. Each CUDA policy requires 
the user to specify the number of threads in each dimension of a thread block.
The total number of blocks needed are determined based on the size of a
loop iteration space and the number of threads per block. As a starting point, 
the following policy may be used with the ``RAJA::forall`` loop

* ``cuda_exec<STRIDE_SIZE>`` where STRIDE_SIZE corresponds to the number of threads in a given block. 

The ``cuda_exec`` policy defines a default thread block size of 256, if no 
argument is provided.

The nested version enables the user to map global threads in the x,y and z 
components via the following execution policies:

* ``cuda_threadblock_x_exec<X_STRIDE_SIZE>`` - Map a loop nest to the block with ``X_STRIDE_SIZE`` threads in the x-component.
* ``cuda_threadblock_y_exec<Y_STRIDE_SIZE>`` - Map a loop nest to the block with ``Y_STRIDE_SIZE`` threads in the y-component.
* ``cuda_threadblock_z_exec<Z_STRIDE_SIZE>`` - Map a loop nest to the block with ``Z_STRIDE_SIZE`` threads in the z-component.

Lastly, under the ``RAJA::nested::forall`` method, the user may also map loop 
nest to blocks and to block local threads using through following policies:

* ``cuda_block_x_exec`` - Map a nested loop level to the x-component of a CUDA thread block.
* ``cuda_block_y_exec`` - Map a nested loop level to the y-component of a CUDA thread block.
* ``cuda_block_z_exec`` - Map a nested loop level to the z-component of a CUDA thread block.

* ``cuda_thread_x_exec`` - Map a nested loop level to the x-component of a block local CUDA thread. 
* ``cuda_thread_y_exec`` - Map a nested loop level to the y-component of a block local CUDA thread. 
* ``cuda_thread_z_exec`` - Map a nested loop level to the z-component of a block local CUDA thread. 
