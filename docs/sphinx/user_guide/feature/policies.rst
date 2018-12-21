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

.. note:: * All RAJA execution policies are in the namespace ``RAJA``.
          * As RAJA functionality is expanded, new policies will be added and
            existing ones may be enabled to work in new ways.

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

----------------------
RAJA::region Policies
----------------------

The following policies may only be used with the ``RAJA::region`` method. 
``RAJA::forall`` and ``RAJA::kernel`` methods may be used within a
``RAJA::region``.

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

-----------------------
RAJA::kernel Policies
-----------------------

The following policies may only be used with the ``RAJA::kernel`` method.

CUDA Policies
^^^^^^^^^^^^^^

* ``cuda_thread_x_direct`` - Direct mapping of loop iterations to cuda threads in the x dimension.
* ``cuda_thread_y_direct`` - Direct mapping of loop iterations to cuda threads in the y dimension.
* ``cuda_thread_z_direct`` - Direct mapping of loop iterations to cuda threads in the z dimension.
  
.. note::  
          * If multiple thread direct policies are used within kernel; the product of the sizes must be :math:`\leq` 1024. 
          * Repeating thread direct policies with the same thread dimension in perfectly nested loops is not supported. 
          * Thread direct policies are only recommended with certain loop patterns such as tiling.

* ``cuda_thread_x_loop`` - Extension to the thread direct policy, introduces a block stride loop based on the thread-block size in the x dimension.
* ``cuda_thread_y_loop`` - Extension to the thread direct policy, introduces a block stride loop based on the thread-block size in the y dimension.
* ``cuda_thread_z_loop`` - Extension to the thread direct policy, introduces a block stride loop based on the thread-block size in the z dimension.

.. note::
          * These polices gives the flexability to have a larger number of iterates than threads in the x/y/z dimension.
          * There is no constraint on the product of sizes of the associated loop iteration space.
          * Cuda thread loop policies are recommended for most loop structures.

* ``cuda_block_x_loop`` - Maps loop iterations to cuda thread blocks in x dimension.
* ``cuda_block_y_loop`` - Maps loop iterations to cuda thread blocks in y dimension.
* ``cuda_block_z_loop`` - Maps loop iterations to cuda thread blocks in z dimension.

.. _loop_elements-kernelpol-label:

--------------------------------
RAJA Kernel Execution Policies
--------------------------------

RAJA kernel policies are constructed with a combination of *Statements* and
*Statement Lists* that forms a simple domain specific language that
relies **solely on standard C++11 template support**. A RAJA Statement is an
action, such as executing a loop, invoking a lambda, setting a thread barrier,
etc. A StatementList is an ordered list of Statements that are composed
to construct a kernel in the order that they appear in the kernel policy.
A Statement may contain an enclosed StatmentList. Thus, a
``RAJA::KernelPolicy`` type is simply a StatementList.

The main Statements types provided by RAJA are ``RAJA::statement::For`` and
``RAJA::statement::Lambda``, that we described above. A 'For' Statement
indicates a for-loop structure and takes three template arguments:
'ArgId', 'ExecPolicy', and 'EnclosedStatements'. The ArgID identifies the
position of the item it applies to in the iteration space tuple argument to the
``RAJA::kernel`` method. The ExecPolicy gives the RAJA execution policy to
use on that loop/iteration space (similar to ``RAJA::forall``).
EnclosedStatements contain whatever is nested within the template parameter
list and form a StatementList, which is executed for each iteration of the loop.
The ``RAJA::statement::Lambda<LambdaID>`` invokes the lambda corresponding to
its position (LambdaID) in the sequence of lambda expressions in the
``RAJA::kernel`` argument list. For example, a simple sequential for-loop::

  for (int i = 0; i < N; ++i) {
    // loop body
  }

would be represented using the RAJA kernel API as::

  using KERNEL_POLICY =
    RAJA::KernelPolicy<
      RAJA::statement::For<0, RAJA::seq_exec,
        RAJA::statement::Lambda<0>
      >
    >;

  RAJA::kernel<KERNEL_POLICY>(
    RAJA::make_tuple(N_range),
    [=](int i) {
      // loop body
    }
  );

The following list summarizes the current collection of ``RAJA::kernel``
statement types:

  * ``RAJA::statement::For< ArgId, ExecPolicy, EnclosedStatements >`` abstracts a for-loop associated with kernel iteration space at tuple index 'ArgId', to be run with 'ExecPolicy' execution policy, and containing the 'EnclosedStatements' which are executed for each loop iteration.

  * ``RAJA::statement::Lambda< LambdaId >`` invokes the lambda expression that appears at position 'LambdaId' in the sequence of lambda arguments.

  * ``RAJA::statement::Collapse< ExecPolicy, ArgList<...>, EnclosedStatements >`` collapses multiple perfectly nested loops specified by tuple iteration space indices in 'ArgList', using the 'ExecPolicy' execution policy, and places 'EnclosedStatements' inside the collapsed loops which are executed for each iteration. Note that this only works for CPU execution policies (e.g., sequential, OpenMP).It may be available for CUDA in the future if such use cases arise.

  * ``RAJA::statement::If< Conditional >`` chooses which portions of a policy to run based on run-time evaluation of conditional statement; e.g., true or false, equal to some value, etc.

  * ``RAJA::statement::CudaKernel< EnclosedStatements>`` launches 'EnclosedStatements' as a CUDA kernel; e.g., a loop nest where the iteration spaces of each loop level are associated with threads and/or thread blocks as described by the execution policies applied to them.

  * ``RAJA::statement::CudaSyncThreads`` provides CUDA '__syncthreads' barrier. Note that a similar thread barrier for OpenMP will be added soon.

  * ``RAJA::statement::Hyperplane< ArgId, HpExecPolicy, ArgList<...>, ExecPolicy, EnclosedStatements >`` provides a hyperplane (or wavefront) iteration pattern over multiple indices. A hyperplane is a set of multi-dimensional index values: i0, i1, ... such that h = i0 + i1 + ... for a given h. Here, 'ArgId' is the position of the loop argument we will iterate on (defines the order of hyperplanes), 'HpExecPolicy' is the execution policy used to iterate over the iteration space specified by ArgId (often sequential), 'ArgList' is a list of other indices that along with ArgId define a hyperplane, and 'ExecPolicy' is the execution policy that applies to the loops in ArgList. Then, for each iteration, everything in the 'EnclosedStatements' is executed.

Various examples that illustrate the use of these statement types can be found
in :ref:`complex_loops-label`.

Additional statement types will be developed to support new use cases as they arise.
