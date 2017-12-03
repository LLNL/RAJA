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

.. _policies-label:

==================
Execution Policies
==================

The following serves as reference to the various policies which ``RAJA`` supports.


.. note:: * All RAJA execution policies are in the namespace ``RAJA``.


--------------------
Serial/SIMD Policies
--------------------

* ``seq_exec``  - Enforces sequential loop iterations.
* ``loop_exec`` - Allows the compiler to vectorize.
* ``simd_exec`` - Introduces vectorization hints to loop iterations.

---------------
OpenMP Policies
---------------

* ``omp_for_exec`` - Distributes loop iterations within threads (static scheduling by default).
* ``omp_for_nowait_exec`` - Removes synchronization within threaded regions.
* ``omp_for_static`` - Distributes loop iterations within threads using a static schedule.
* ``omp_parallel_exec`` - Creates a parallel region.
* ``omp_parallel_for_exec`` - Creates a parallel region and divides loop iterations between threads.
* ``omp_parallel_segit`` - Creates a parallel region for index segments.
* ``omp_parallel_for_segit`` - Create a parallel region for index segments and divide segments between threads.
* ``omp_collapse_nowait_exec`` - Collapses multiple iteration spaces into a single space and removes any implied barriers.

----------------------
OpenMP Target Policies
----------------------

* ``omp_target_parallel_for_exec`` - Variables are mapped to a device environment and a parallel region is created in which loop iterations are divided among threads.

------------
TBB Policies
------------

* ``tbb_for_exec`` - Schedules tasks to operate in parallel using the static scheduler.
* ``tbb_for_static`` - Implements the parallel_for method and uses a static scheduler.
* ``tbb_for_dynamic`` - Implements the parallel_for method and uses a dynamic scheduler.
* ``tbb_segit`` - Implements the parallel_for for indexset segments.

-------------
CUDA Policies
-------------

Following the CUDA nomenclature, GPU computations are performed on a predefined compute grid.
Each unit of the grid is referred to as a thread and threads are furthered grouped into
thread blocks. Threads and thread blocks may have up to three-dimensional indexing for convenience.
Each CUDA policy requires the user to specify the number of threads in each dimension of a thread block.
The total number of blocks needed are determined based on the iteration space and the number of threads
per block. As a starting point, the following policy may be used with the ``RAJA::forall`` loop

* ``cuda_exec<int STRIDE_SIZE>`` where STRIDE_SIZE corresponds to the number of threads in a given block.

The nested version enables the user to map global threads in the x,y and z components via the following
execution policies

* ``cuda_threadblock_x_exec<int X_STRIDE_SIZE>`` - Maps a loop nest to the block with ``X_STRIDE_SIZE`` threads in the x-component.
* ``cuda_threadblock_y_exec<int Y_STRIDE_SIZE>`` - Maps a loop nest to the block with ``Y_STRIDE_SIZE`` threads in the y-component.
* ``cuda_threadblock_z_exec<int Z_STRIDE_SIZE>`` - Maps a loop nest to the block with ``Z_STRIDE_SIZE`` threads in the z-component.

Lastly, under the ``RAJA::nested::forall`` method, the user also map loop nest to blocks and threads
using through following policies

* ``cuda_block_x_exec`` - Maps a loop nest to blocks in the x-component.
* ``cuda_block_y_exec`` - Maps a loop nest to blocks in the y-component.
* ``cuda_block_z_exec`` - Maps a loop nest to blocks in the z-component.

* ``cuda_thread_x_exec`` - Maps a loop nest to threads in the x-component.
* ``cuda_thread_y_exec`` - Maps a loop nest to threads in the y-component.
* ``cuda_thread_z_exec`` - Maps a loop nest to threads in the z-component.
