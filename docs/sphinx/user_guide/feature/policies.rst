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
and ``RAJA::scan`` methods.

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
* ``simd_exec`` - Attempt to force SIMD vectorization of a loop via vectorization compiler hints inside the RAJA internal implementation.
* ``loop_exec`` - Allow the compiler to generate whichever optimizations, such as SIMD, that it thinks are appropriate; effectively an undecorated loop with no
pragmas or intrinsics to prevent or encourage compiler optimizations.

OpenMP Policies
^^^^^^^^^^^^^^^^

* ``omp_parallel_for_exec`` - Execute a loop in parallel by creating an OpenMP parallel region and distribute loop iterations across threads within it; i.e., use an ``omp parallel for`` pragma in the RAJA implementation.
* ``omp_for_exec`` - Execute a loop in parallel using an ``omp for`` pragma within an existing parallel region. 
* ``omp_for_static<CHUNK_SIZE>`` - Execute a loop in parallel using a static schedule with given chunk size within an existing parallel region; i.e., use an ``omp parallel for schedule(static, CHUNK_SIZE>`` pragma.
* ``omp_for_nowait_exec`` - Execute loop in an existing parallel region without synchronization after the loop; i.e., use an ``omp for nowait`` clause.

.. note:: To control the number of OpenMP threads used by these policies:
          set the value of the environment variable 'OMP_NUM_THREADS' (which is
          fixed for duration of run), or call the OpenMP routine 
          'omp_set_num_threads(nthreads)' (which allows changing number of 
          threads at runtime).

OpenMP Target Policies
^^^^^^^^^^^^^^^^^^^^^^^^
* ``omp_target_parallel_for_exec<NUMTEAMS>`` - Execute a loop in parallel using an ``omp teams distribute parallel for num_teams(NUMTEAMS)`` pragma with given number of thread teams inside a ``omp target`` region; e.g., if a GPU device is available, this is similar to launching a CUDA kernel with 
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
index set execution policy is required. An 
index set execution policy is a **two-level policy**: an 'outer' policy for 
iterating over segments in the index set, and an 'inner' policy used to
execute the iterations defined by each segment. An index set execution policy 
type has the form::

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
``RAJA::forall`` and ``RAJA::kernel`` methods may be used within a parallel
region created with the ``RAJA::region`` construct.

* ``seq_region`` - Create a sequential region (see note below).
* ``omp_parallel_region`` - Create an OpenMP parallel region.

For example, the following code will execute two consecutive loops in parallel in an OpenMP parallel region without thread synchronization between them::

  RAJA::region<RAJA::omp_parallel_region>( [=]() {

    RAJA::forall<RAJA::omp_for_nowait_exec>(
      RAJA::RangeSegment(0, N), [=](int i) {
        // loop body #1
    });

    RAJA::forall<RAJA::omp_for_nowait_exec>(
      RAJA::RangeSegment(0, N), [=](int i) {
        // loop body #2
    });

  }); // end omp parallel region

.. note:: The sequential region does not really do anything useful. It is 
          provided so that, if you want to turn off OpenMP in your code, 
          you can simply replace the region policy type and you do not
          have to change your source code. 

-------------------------
RAJA::scan Policies
-------------------------

Generally, any execution policy that works with ``RAJA::forall`` methods will 
also work with ``RAJA::scan`` methods. See :ref:`scan-label` for information
about RAJA scan methods.

-------------------------
Reduction Policies
-------------------------

Each RAJA reduction object must be defined with a 'reduction policy'
type. Reduction policy types are distinct from loop execution policy types.
A reduction policy type must be consistent with the execution policy in the
kernel where the reduction is used. See :ref:`reductions-label` for more 
information.

-------------------------
Local Array Policies
-------------------------

``RAJA::LocalArray`` types must use a memory allocation policy indicating
where the memory for the local array will live. These policies are described
in :ref:`local_array-label`.


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
    * Repeating thread direct policies with the same thread dimension in perfectly nested loops is not recommended. Your code may do something, but likely will not do what you expect and/or be correct. 
    * If multiple thread direct policies are used in a kernel (using different thread dimensions), the product of sizes of the corresponding iteration spaces must be :math:`\leq` 1024. You cannot launch a CUDA kernel with more than 1024 threads per block.
    * **Thread direct policies are only recommended with certain loop patterns, such as tiling.**

* ``cuda_thread_x_loop`` - Extension to the thread direct policy by introducing a block stride loop based on the thread-block size in the x dimension.
* ``cuda_thread_y_loop`` - Extension to the thread direct policy by introducing a block stride loop based on the thread-block size in the y dimension.
* ``cuda_thread_z_loop`` - Extension to the thread direct policy by introducing a block stride loop based on the thread-block size in the z dimension.

.. note::
    * There is no constraint on the product of sizes of the associated loop iteration space.
    * These polices enable a having a larger number of iterates than threads in the x/y/z thread dimension.
    * **Cuda thread loop policies are recommended for most loop patterns.**

* ``cuda_block_x_loop`` - Maps loop iterations to cuda thread blocks in x dimension.
* ``cuda_block_y_loop`` - Maps loop iterations to cuda thread blocks in y dimension.
* ``cuda_block_z_loop`` - Maps loop iterations to cuda thread blocks in z dimension.

OpenMP Target Policies
^^^^^^^^^^^^^^^^^^^^^^^

* ``omp_target_parallel_collapse_exec`` - Collapse specified loops and execute kernel in OpenMP target region; i.e., apply ``omp teams distribute parallel for collapse(...)`` pragma with given number of loops to collapse inside a ``omp target`` region.

.. _loop_elements-kernelpol-label:

--------------------------------
RAJA Kernel Execution Policies
--------------------------------

RAJA kernel execution policy constructs form a simple domain specific language 
for composing and transforming complex loops that relies 
**solely on standard C++11 template support**. 
RAJA kernel policies are constructed using a combination of *Statements* and
*Statement Lists*. A RAJA Statement is an action, such as execute a loop, 
invoke a lambda, set a thread barrier, etc. A StatementList is an ordered list 
of Statements that are composed in the order that they appear in the kernel 
policy to construct a kernel. A Statement may contain an enclosed StatmentList. Thus, a ``RAJA::KernelPolicy`` type is really just a StatementList.

The main Statements types provided by RAJA are ``RAJA::statement::For`` and
``RAJA::statement::Lambda``, that we have shown above. A 'For' Statement
indicates a for-loop structure and takes three template arguments:
'ArgId', 'ExecPolicy', and 'EnclosedStatements'. The ArgID identifies the
position of the item it applies to in the iteration space tuple argument to the
``RAJA::kernel`` method. The ExecPolicy is the RAJA execution policy to
use on that loop/iteration space (similar to ``RAJA::forall``).
EnclosedStatements contain whatever is nested within the template parameter
list to form a StatementList, which will be executed for each iteration of 
the loop. The ``RAJA::statement::Lambda<LambdaID>`` invokes the lambda 
corresponding to its position (LambdaID) in the sequence of lambda expressions 
in the ``RAJA::kernel`` argument list. For example, a simple sequential 
for-loop::

  for (int i = 0; i < N; ++i) {
    // loop body
  }

can be represented using the RAJA kernel interface as::

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

.. note:: All ``RAJA::forall`` functionality can be done using the 
          ``RAJA::kernel`` interface. We maintain the ``RAJA::forall``
          interface since it is less verbose and thus more convenient
          for users.
   
RAJA::kernel Statement Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The list below summarizes the current collection of statement types that
can be used with ``RAJA::kernel`` and ``RAJA::kernel_param``. More detailed
explanation along with examples of how they are used can be found in 
:ref:`tutorial-label`.

.. note:: * All of these statement types are in the namespace ``RAJA``.
          * ``RAJA::kernel_param`` functions similar to ``RAJA::kernel`` except             that its second argument is a *tuple of parameters* used in a kernel
            for local arrays, thread local variables, tiling information, etc.

  * ``statement::For< ArgId, ExecPolicy, EnclosedStatements >`` abstracts a for-loop associated with kernel iteration space at tuple index 'ArgId', to be run with 'ExecPolicy' execution policy, and containing the 'EnclosedStatements' which are executed for each loop iteration.

  * ``statement::Lambda< LambdaId >`` invokes the lambda expression that appears at position 'LambdaId' in the sequence of lambda arguments.

  * ``statement::Collapse< ExecPolicy, ArgList<...>, EnclosedStatements >`` collapses multiple perfectly nested loops specified by tuple iteration space indices in 'ArgList', using the 'ExecPolicy' execution policy, and places 'EnclosedStatements' inside the collapsed loops which are executed for each iteration. Note that this only works for CPU execution policies (e.g., sequential, OpenMP).It may be available for CUDA in the future if such use cases arise.

  * ``statement::CudaKernel< EnclosedStatements>`` launches 'EnclosedStatements' as a CUDA kernel; e.g., a loop nest where the iteration spaces of each loop level are associated with threads and/or thread blocks as described by the execution policies applied to them.

  * ``statement::CudaSyncThreads`` provides CUDA '__syncthreads' barrier. Note that a similar thread barrier for OpenMP will be added soon.

  * ``statement::InitLocalMem< MemPolicy, ParamList<...>, EnclosedStatements >`` allocates memory for a ``RAJA::LocalArray`` object used in kernel. The 'ParamList' entries indicate which local array objects in a tuple will be initialized. The 'EnclosedStatements' contain the code in which the local array will be accessed; e.g., initialization operations.

  * ``statement::Tile< ArgId, TilePolicy, ExecPolicy, EnclosedStatements >`` abstracts an outer tiling loop containing an inner for-loop over each tile. The 'ArgId' indicates which entry in the iteration space tuple to which the tiling loop applies and the 'TilePolicy' specifies the tiling pattern to use, including its dimension. The 'ExecPolicy' and 'EnclosedStatements' are similar to what they represent in a ``statement::For`` type.

  * ``statement::TileTCount< ArgId, ParamId, TilePolicy, ExecPolicy, EnclosedStatements >`` abstracts an outer tiling loop containing an inner for-loop over each tile, **where it is necessary to obtain the tile number in each tile**. The 'ArgId' indicates which entry in the iteration space tuple to which the loop applies and the 'ParamId' indicates the position of the tile number in the parameter tuple. The 'TilePolicy' specifies the tiling pattern to use, including its dimension. The 'ExecPolicy' and 'EnclosedStatements' are similar to what they represent in a ``statement::For`` type.

  * ``statement::tile_fixed<TileSize>`` partitions loop iterations into tiles of a fixed size specified by 'TileSize'. This statement type can be used as the 'TilePolicy' template paramter in the Tile statements above.

  * ``statement::ForICount< ArgId, ParamId, ExecPolicy, EnclosedStatements >`` abstracts an inner for-loop within an outer tiling loop **where it is necessary to obtain the local iteration index in each tile**. The 'ArgId' indicates which entry in the iteration space tuple to which the loop applies and the 'ParamId' indicates the position of the tile index parameter in the parameter tuple. The 'ExecPolicy' and 'EnclosedStatements' are similar to what they represent in a ``statement::For`` type.

  * ``statement::If< Conditional >`` chooses which portions of a policy to run based on run-time evaluation of conditional statement; e.g., true or false, equal to some value, etc.

  * ``statement::Hyperplane< ArgId, HpExecPolicy, ArgList<...>, ExecPolicy, EnclosedStatements >`` provides a hyperplane (or wavefront) iteration pattern over multiple indices. A hyperplane is a set of multi-dimensional index values: i0, i1, ... such that h = i0 + i1 + ... for a given h. Here, 'ArgId' is the position of the loop argument we will iterate on (defines the order of hyperplanes), 'HpExecPolicy' is the execution policy used to iterate over the iteration space specified by ArgId (often sequential), 'ArgList' is a list of other indices that along with ArgId define a hyperplane, and 'ExecPolicy' is the execution policy that applies to the loops in ArgList. Then, for each iteration, everything in the 'EnclosedStatements' is executed.

Various examples that illustrate the use of these statement types can be found
in :ref:`complex_loops-label`.
