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

This section describes the execution policies that ``RAJA`` provides and 
indicates which policies may be used with ``RAJA::forall``, ``RAJA::kernel``,
and/or ``RAJA::scan`` methods.

.. note:: All RAJA execution policies are in the namespace ``RAJA``.

.. note:: As RAJA functionality is expanded, new policies may be added and
          existing ones may be enabled to work with other RAJA loop constructs.

-----------------------------------------------------
RAJA::forall and RAJA::kernel Policies
-----------------------------------------------------

The following list of policies may be used with either ``RAJA::forall`` and
``RAJA::kernel`` methods.

Serial/SIMD Policies
^^^^^^^^^^^^^^^^^^^^^^

* ``seq_exec``  - Strictly sequential loop execution.
* ``simd_exec`` - Forced SIMD execution by adding vectorization hints.
* ``loop_exec`` - Allows the compiler to generate whichever optimizations (e.g., SIMD) that it thinks are appropriate.

OpenMP Policies
^^^^^^^^^^^^^^^^

* ``omp_parallel_for_exec`` - Execute a loop in parallel using an ``omp parallel for`` pragma; i.e., create a parallel region and distribute loop iterations across threads.
* ``omp_for_exec`` - Execute a loop in parallel using an ``omp for`` pragma within an exiting parallel region. 
* ``omp_for_static<CHUNK_SIZE>`` - Execute a loop in parallel using a static schedule with given chunk size within an existing parallel region; i.e., use an ``omp parallel for schedule(static, CHUNK_SIZE>`` pragma.
* ``omp_for_nowait_exec`` - Execute loop in an existing parallel region without synchronization after the loop; i.e., use an ``omp for nowait`` clause.

.. note:: To control the number of OpenMP threads used by these policies:
          set the value of the environment variable 'OMP_NUM_THREADS' (which is
          fixed for duration of run), or call the OpenMP routine 
          'omp_set_num_threads(nthreads)' (which allows changing number of 
          threads at runtime).

OpenMP Target Policies
^^^^^^^^^^^^^^^^^^^^^^^^
* ``omp_target_parallel_for_exec<NUMTEAMS>`` - Execute a loop in parallel using an ``omp target parallel for`` pragma with given number of thread teams; e.g.,
if a GPU device is available, this is similar to launching a CUDA kernel with 
a thread block size of NUMTEAMS. 

Intel Threading Building Blocks (TBB) Policies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``tbb_for_exec`` - Schedule loop iterations as tasks to execute in parallel using a TBB ``parallel_for`` method.
* ``tbb_for_static<CHUNK_SIZE>`` - Schedule loop iterations as tasks to execute in parallel using a TBB ``parallel_for`` method with a static partitioner using given chunk size.
* ``tbb_for_dynamic`` - Schedule loop iterations as tasks to execute in parallel using a TBB ``parallel_for`` method with a dynamic scheduler.

.. note:: To control the number of TBB worker threads used by these policies:
          set the value of the environment variable 'TBB_NUM_WORKERS' (which is
          fixed for duration of run), or create a 'task_scheduler_init' object::

            tbb::task_scheduler_init TBBinit( nworkers );

            // do some parallel work

            TBBinit.terminate();
            TBBinit.initialize( new_nworkers );

            // do some more parallel work

          This allows changing number of workers at runtime.

-------------------------------
RAJA::forall Policies
-------------------------------

The following list of policies may only be used with a ``RAJA::forall`` method.

CUDA Policies 
^^^^^^^^^^^^^^^^^^

* ``cuda_exec<BLOCK_SIZE>`` - Execute a loop in a CUDA kernel launched with given thread block size. If no thread block size is given, a default size of 256 is used.

IndexSet Policies
^^^^^^^^^^^^^^^^^^

When a ``RAJA::forall`` method is used with a ``RAJA::IndexSet`` object, an
index set execution policy must be used to guarantee correct behavior. An 
index set execution policy is a two-level policy: an 'outer' policy for 
iterating over segments in the index set, and an 'inner' policy for executing
each segment. An index set execution policy type has the form::

  RAJA::ExecPolicy< segment_iteration_policy, segment_execution_policy>

See :ref:`indexsets-label` for more information.

Generally, any policy that can be used with a ``RAJA::forall`` method
can be used as the segment execution policy. The following policies are
available to use for the segment iteration policy:

* ``seq_segit`` - Iterate over index set segments sequentially.
* ``omp_parallel_segit`` - Iterate over index set segments in parallel using an OpenMP parallel loop.
* ``omp_parallel_for_segit`` - Same as above.
* ``tbb_segit`` - Iterate over an index set segments in parallel using a TBB 'parallel_for' method.

-----------------------
RAJA::kernel Policies
-----------------------

The following policies may only be used with the ``RAJA::kernel`` method.

CUDA Policies
^^^^^^^^^^^^^^

* ``cuda_block_exec`` - Map loop iterations to CUDA thread blocks.
* ``cuda_thread_exec`` - Map loop iterations to CUDA threads in a thread block.
* ``cuda_threadblock_exec<BLOCK_SIZE>`` - Map loop iterations to CUDA thread blocks, each with given block size number of threads.

----------------------
RAJA::region Policies
----------------------

The following policies may only be used with the ``RAJA::region`` method.

* ``seq_region_exec`` - Creates a sequential region.
* ``omp_parallel_region_exec`` - Create an OpenMP parallel region.

-------------------------
RAJA::scan Policies
-------------------------

Generally, any execution policy that works with ``RAJA::forall`` methods will 
also work with ``RAJA::scan`` methods. See :ref:`scan-label` for information
about RAJA scan methods.

-------------------------
RAJA Reduction Policies
-------------------------

Note that a RAJA reduction object must be defined with a 'reduction policy'
type. Reduction policy types are distinct from loop execution policy types.
A reduction policy type must be consistent with the loop execution policy
that is used. See :ref:`reductions-label` for more information.
