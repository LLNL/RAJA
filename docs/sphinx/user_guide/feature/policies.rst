.. ##
.. ## Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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
Policies
==================

This section describes various RAJA policies for loop kernel execution,
scans, reductions, atomics, etc. Each policy is a type that is passed to
a RAJA template method or class to specialize its behavior. Typically, the
policy indicates which programming model back-end to use and sometimes
provides additional information about the execution pattern, such as
number of CUDA threads per threadblock, whether execution is synchronous
or asynchronous, etc.

As RAJA functionality is expanded, new policies will be added and some may
be redefined and to work in new ways.

.. note:: * All RAJA policies are in the namespace ``RAJA``.

-----------------------------------------------------
RAJA Loop/Kernel Execution Policies
-----------------------------------------------------

The following table summarizes RAJA policies for executing loops and kernels.

====================================== ============= ===========================
Execution Policy                       Works with    Brief description
====================================== ============= ===========================
**Sequential/SIMD**
seq_exec                               forall,       Strictly sequential
                                       kernel (For), execution
                                       scans
simd_exec                              forall,       Try to force generation of
                                       kernel (For), SIMD instructions via
                                       scans         compiler hints in RAJA
                                                     internal implementation
loop_exec                              forall,       Allow compiler to generate
                                       kernel (For), any optimizations, such as
                                       scans         SIMD, that may be
                                                     beneficial according to
                                                     its heuristics;
                                                     i.e., no loop decorations
                                                     (pragmas or intrinsics) in
                                                     RAJA implementation
**OpenMP CPU multithreading**
(see note below table)
omp_parallel_for_exec                  forall,       Create OpenMP parallel
                                       kernel (For), region and execute with CPU
                                       scans         multithreading inside it;
                                                     i.e., apply ``omp parallel
                                                     for`` pragma on loop
omp_for_exec                           forall,       Parallel execution with
                                       kernel (For)  OpenMP CPU multithreading
                                                     inside an *existing* 
                                                     parallel region; i.e., 
                                                     apply ``omp for`` pragma 
                                                     on loop
omp_for_static<CHUNK_SIZE>             forall,       Execute loop with OpenMP
                                       kernel (For)  CPU multithreading using
                                                     static schedule and given
                                                     chunk size inside an 
                                                     *existing* parallel region;
                                                     i.e., apply ``omp for  
                                                     schedule(static, 
                                                     CHUNK_SIZE>`` pragma on 
                                                     loop 
omp_for_nowait_exec                    forall,       Parallel execution with
                                       kernel (For)  OpenMP CPU multithreading
                                                     inside an existing parallel
                                                     region without
                                                     synchronization after loop;
                                                     i.e., apply
                                                     ``omp for nowait`` pragma
**Intel Threading Building Blocks**
(see note below table)
tbb_for_exec                           forall,       Execute loop iterations
                                       kernel (For), as tasks in parallel using
                                       scans         TBB ``parallel_for`` method
tbb_for_static<CHUNK_SIZE>             forall,       Same as above, but use
                                       kernel (For), a static scheduler with
                                       scans         given chunk size
tbb_for_dynamic                        forall,       Same as above, but use
                                       kernel (For), a dynamic scheduler
                                       scans 
**CUDA** 
(see notes below table)
cuda_exec<BLOCK_SIZE>                  forall,       Execute loop iterations
                                       kernel (For), in a CUDA kernel launched
                                       scans         with given thread-block
                                                     size. If none given, use
                                                     default value of 256 
                                                     threads/block 
cuda_thread_x_direct                   kernel (For)  Map loop iterations to CUDA
                                                     threads in x-dimension
cuda_thread_y_direct                   kernel (For)  Map loop iterations to CUDA
                                                     threads in y-dimension
cuda_thread_z_direct                   kernel (For)  Map loop iterations to CUDA
                                                     threads in z-dimension
cuda_thread_x_loop                     kernel (For)  Extends thread-x-direct
                                                     policy by adding a 
                                                     block-stride loop
cuda_thread_y_loop                     kernel (For)  Extends thread-y-direct
                                                     policy by adding a 
                                                     block-stride loop
cuda_thread_z_loop                     kernel (For)  Extends thread-z-direct
                                                     policy by adding a 
                                                     block-stride loop
cuda_block_x_loop                      kernel (For)  Map loop iterations to CUDA
                                                     thread blocks in 
                                                     x-dimension
cuda_block_y_loop                      kernel (For)  Map loop iterations to CUDA
                                                     thread blocks in 
                                                     y-dimension
cuda_block_z_loop                      kernel (For)  Map loop iterations to CUDA
                                                     thread blocks in
                                                     z-dimension
**OpenMP target**
omp_target_parallel_for_exec<NUMTEAMS> forall        Create parallel target 
                                                     region and execute with 
                                                     given number of thread 
                                                     teams inside it; i.e.,
                                                     apply ``omp teams 
                                                     distribute parallel for 
                                                     num_teams(NUMTEAMS)`` 
                                                     pragma on loop 
omp_target_parallel_collapse_exec      kernel        Similar to above, but 
                                       (Collapse)    collapse *perfectly-nested*                                                     loops, which are specified
                                                     in arguments to RAJA
                                                     Collapse statement. Note:
                                                     compiler determines number
                                                     of thread teams and threads
                                                     per team
====================================== ============= ===========================

The following notes apply to the execution policies described in the table 
above.

.. note:: To control the number of threads used by OpenMP policies
          set the value of the environment variable 'OMP_NUM_THREADS' (which is
          fixed for duration of run), or call the OpenMP routine 
          'omp_set_num_threads(nthreads)' (which allows changing number of 
          threads at runtime).

.. note:: To control the number of worker threads used by TBB policies:
          set the value of the environment variable 'TBB_NUM_WORKERS' (which is
          fixed for duration of run), or create a 'task_scheduler_init' object::

            tbb::task_scheduler_init TBBinit( nworkers );

            // do some parallel work

            TBBinit.terminate();
            TBBinit.initialize( new_nworkers );

            // do some more parallel work

          This allows changing number of workers at runtime.

Several notable constraints apply to RAJA CUDA thread-direct policies.

.. note:: * Repeating thread direct policies with the same thread dimension in perfectly nested loops is not recommended. Your code may do something, but likely will not do what you expect and/or be correct.
          * If multiple thread direct policies are used in a kernel (using different thread dimensions), the product of sizes of the corresponding iteration spaces must be :math:`\leq` 1024. You cannot launch a CUDA kernel with more than 1024 threads per block.
          * **Thread-direct policies are recommended only for certain loop patterns, such as tiling.**

Several notes regarding CUDA thread and block loop policies are also good to 
know.

.. note:: * There is no constraint on the product of sizes of the associated loop iteration space.
          * These polices allow having a larger number of iterates than threads in the x, y, or z thread dimension.
          * **Cuda thread and block loop policies are recommended for most loop patterns.**

.. _indexsetpolicy-label:

-----------------------------------------------------
RAJA IndexSet Execution Policies
-----------------------------------------------------

When an IndexSet iteration space is used in RAJA, such as passing an IndexSet
to a ``RAJA::forall`` method, an index set execution policy is required. An
index set execution policy is a **two-level policy**: an 'outer' policy for
iterating over segments in the index set, and an 'inner' policy used to
execute the iterations defined by each segment. An index set execution policy
type has the form::

  RAJA::ExecPolicy< segment_iteration_policy, segment_execution_policy>

See :ref:`indexsets-label` for more information.

In general, any policy that can be used with a ``RAJA::forall`` method
can be used as the segment execution policy. The following policies are
available to use for the segment iteration policy:

====================================== =========================================
Execution Policy                       Brief description
====================================== =========================================
**Serial**
seq_segit                              Iterate over index set segments 
                                       sequentially
**OpenMP CPU multithreading**          
omp_parallel_segit                     Create OpenMP parallel region and 
                                       iterate over segments in parallel inside                                        it; i.e., apply ``omp parallel for`` 
                                       pragma on loop over segments
omp_parallel_for_segit                 Same as above
**Intel Threading Building Blocks**
tbb_segit                              Iterate over index set segments in 
                                       parallel using a TBB 'parallel_for' 
                                       method
====================================== =========================================

-------------------------
Parallel Region Policies
-------------------------

The following policies may only be used with the ``RAJA::region`` method. 
``RAJA::forall`` and ``RAJA::kernel`` methods may be used within a parallel
region created with the ``RAJA::region`` construct.

* ``seq_region`` - Create a sequential region (see note below).
* ``omp_parallel_region`` - Create an OpenMP parallel region.

For example, the following code will execute two consecutive loops in parallel 
in an OpenMP parallel region without synchronizing threads between them::

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

.. note:: The sequential region specialization is essentially a *pass through*
          operation. It is provided so that if you want to turn off OpenMP in 
          your code, you can simply replace the region policy type and you do 
          not have to change your algorithm source code. 

.. _reducepolicy-label:

-------------------------
Reduction Policies
-------------------------

Each RAJA reduction object must be defined with a 'reduction policy'
type. Reduction policy types are distinct from loop execution policy types.
It is important to note the following constraints about RAJA reduction usage:

.. note:: To guarantee correctness, a **reduction policy must be consistent
          with the loop execution policy** used. For example, a CUDA
          reduction policy must be used when the execution policy is a
          CUDA policy, an OpenMP reduction policy must be used when the
          execution policy is an OpenMP policy, and so on.

The following table summarizes RAJA reduction policy types:

===================== ============= ===========================================
Reduction Policy      Loop Policies Brief description
                      to Use With
===================== ============= ===========================================
seq_reduce            seq_exec,     Non-parallel (sequential) reduction
                      loop_exec 
omp_reduce            any OpenMP    OpenMP parallel reduction
                      policy
omp_reduce_ordered    any OpenMP    OpenMP parallel reduction with result
                      policy        guaranteed to be reproducible
omp_target_reduce     any OpenMP    OpenMP parallel target offload reduction
                      target policy
tbb_reduce            any TBB       TBB parallel reduction
                      policy
cuda_reduce           any CUDA      Parallel reduction in a CUDA kernel
                      policy        (device synchronization will occur when 
                                    reduction value is finalized)
cuda_reduce_atomic    any CUDA      Same as above, but reduction may use CUDA
                      policy        atomic operations
===================== ============= ===========================================

.. note:: RAJA reductions used with SIMD execution policies are not
          guaranteed to generate correct results at present.

.. _atomicpolicy-label:

-------------------------
Atomic Policies
-------------------------

Each RAJA atomic operation must be defined with an 'atomic policy'
type. Atomic policy types are distinct from loop execution policy types.

.. note :: An atomic policy type must be consistent with the loop execution 
           policy for the kernel in which the atomic operation is used. The
           following table summarizes RAJA atomic policies and usage.

===================== ============= ===========================================
Atomic Policy         Loop Policies Brief description
                      to Use With
===================== ============= ===========================================
seq_atomic            seq_exec,     Atomic operation performed in a non-parallel
                      loop_exec     (sequential) kernel
omp_atomic            any OpenMP    Atomic operation performed in an OpenMP 
                      policy        multithreading or target kernel; i.e., 
                                    apply ``omp atomic`` pragma
cuda_atomic           any CUDA      Atomic operation performed in a CUDA kernel
                      policy        
builtin_atomic        seq_exec,     Compiler *builtin* atomic operation
                      loop_exec,
                      any OpenMP
                      policy        
auto_atomic           seq_exec,     Atomic operation *compatible* with loop
                      loop_exec,    execution policy. See example below.
                      any OpenMP
                      policy,
                      any CUDA
                      policy                 
===================== ============= ===========================================

Here is an example illustrating use of the ``auto_atomic`` policy::

  RAJA::forall< RAJA::cuda_exec >(RAJA::RangeSegment seg(0, N),
    [=] RAJA_DEVICE (RAJA::Index_type i) {

    RAJA::atomic::atomicAdd< RAJA::auto_atomic >(&sum, 1);

  });

In this case, the atomic operation knows that it is used in a CUDA kernel
context and the CUDA atomic operation is applied. Similarly, if an OpenMP 
execution policy was used, the OpenMP version of the atomic operation would 
be used.

.. note:: * There are no RAJA atomic policies for TBB (Intel Threading Building
            Blocks) execution contexts at present.
          * The ``builtin_atomic`` policy may be preferable to the 
            ``omp_atomic`` policy in terms of performance.

.. _localarraypolicy-label:

----------------------------
Local Array Memory Policies
----------------------------

``RAJA::LocalArray`` types must use a memory policy indicating
where the memory for the local array will live. These policies are described
in :ref:`local_array-label`.

The following memory policies are available to specify memory allocation
for ``RAJA::LocalArray`` objects:

  *  ``RAJA::cpu_tile_mem`` - Allocate CPU memory on the stack
  *  ``RAJA::cuda_shared_mem`` - Allocate CUDA shared memory
  *  ``RAJA::cuda_thread_mem`` - Allocate CUDA thread private memory


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

The main Statement types provided by RAJA are ``RAJA::statement::For`` and
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

  * ``RAJA::statement::Reduce< ReducePolicy, Operator, ParamId, EnclosedStatements >`` reduces a value across threads to a single thread. The 'ReducePolicy' is similar to what it represents for RAJA reduction types. 'ParamId' specifies the position of the reduction value in the parameter tuple passed to the ``RAJA::kernel_param`` method. 'Operator' is the binary operator used in the reduction; typically, this will be one of the operators that can be used with RAJA scans (see :ref:`scanops-label`. After the reduction is complete, the 'EnclosedStatements' execute on the thread that received the final reduced value.

  * ``statement::If< Conditional >`` chooses which portions of a policy to run based on run-time evaluation of conditional statement; e.g., true or false, equal to some value, etc.

  * ``statement::Hyperplane< ArgId, HpExecPolicy, ArgList<...>, ExecPolicy, EnclosedStatements >`` provides a hyperplane (or wavefront) iteration pattern over multiple indices. A hyperplane is a set of multi-dimensional index values: i0, i1, ... such that h = i0 + i1 + ... for a given h. Here, 'ArgId' is the position of the loop argument we will iterate on (defines the order of hyperplanes), 'HpExecPolicy' is the execution policy used to iterate over the iteration space specified by ArgId (often sequential), 'ArgList' is a list of other indices that along with ArgId define a hyperplane, and 'ExecPolicy' is the execution policy that applies to the loops in ArgList. Then, for each iteration, everything in the 'EnclosedStatements' is executed.

Examples that show how to use a variety of these statement types can be found
in :ref:`tutorialcomplex-label`.
