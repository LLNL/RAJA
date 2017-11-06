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

* ``seq_exec``  - Strictly sequential execution.
* ``loop_exec`` - Allow compiler to optimize using SIMD vectorization if it can, but don't use special compiler hints to do so.
* ``simd_exec`` - Apply compiler-specific compiler hints for SIMD optimizations.

---------------
OpenMP Policies
---------------

* ``omp_for_exec`` - Distributes loop iterations within threads
* ``omp_for_nowait_exec`` - Removes synchronization within threaded regions
* ``omp_for_static`` - Assigns each thread approximately the same number of iterations
* ``omp_parallel_exec`` - Creates a parallel region
* ``omp_parallel_for_exec`` - Creates a parallel region and divide loop iterations between threads
* ``omp_parallel_segit`` - Creates a parallel region for index segments
* ``omp_parallel_for_segit`` - Create a parallel region for index segments and divide segments between threads
* ``omp_collapse_nowait_exec`` - Collapses multiple iteration spaces into a single space and removes any implied barries

----------------------
OpenMP Target Policies
----------------------

* ``omp_target_parallel_for_exec`` - Maps variables to a device environment and create parallel region dividing loop iterations between threads
  
------------
TBB Policies
------------ 

* ``tbb_for_exec`` - Schedules tasks to operate in parallel using the static scheduler
* ``tbb_for_static`` - Implements the parallel_for method and uses a static scheduler 
* ``tbb_for_dynamic`` - Implements the parallel_for method and uses a dynamic scheduler
* ``tbb_segit`` - Implements the parallel_for for indexset segments 

-------------
CUDA Policies
-------------

Following the CUDA nomenclature, computations are performed on a predefined compute grid.
Each unit of the grid is referred to as a thread and threads are furthered grouped into 
thread blocks. Threads and thread blocks may have up to three-dimensional indexing for convinence.

Each CUDA policy requires the user to specify the number of threads in each dimension of a thread block. 
The total number of blocks needed are determined based on a the iteration space and the number of threads
per block.

* ``cuda_threadblock_x_exec<int T_x>`` - Constructs a thread block with ``T_x`` threads in the x-component
* ``cuda_threadblock_y_exec<int T_y>`` - Constructs a thread block with ``T_y`` threads in the y-component
* ``cuda_threadblock_z_exec<int T_z>`` - Constructs a thread block with ``T_z`` threads in the z-component