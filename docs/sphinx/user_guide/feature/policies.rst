.. ##
.. ## Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
.. ## and other RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _feat-policies-label:

==================
Policies
==================

RAJA kernel execution methods take an execution policy type template parameter
to specialize execution behavior. Typically, the policy indicates which 
programming model back-end to use and other information about the execution 
pattern, such as number of CUDA threads per thread block, whether execution is 
synchronous or asynchronous, etc. This section describes RAJA policies for 
loop kernel execution, scans, sorts, reductions, atomics, etc. Please
detailed examples in :ref:`tutorial-label` for a variety of use cases.

As RAJA functionality evolves, new policies are added and some may
be redefined and to work in new ways.

.. note:: * All RAJA policies are in the namespace ``RAJA``.
          * All RAJA policies have a prefix indicating the back-end 
            implementation that they use; e.g., ``omp_`` for OpenMP, ``cuda_``
            for CUDA, etc.

-----------------------------------------------------
RAJA Loop/Kernel Execution Policies
-----------------------------------------------------

The following tables summarize RAJA policies for executing kernels.
Please see notes below policy descriptions for additional usage details and
caveats.


Sequential CPU Policies
^^^^^^^^^^^^^^^^^^^^^^^^

For the sequential CPU back-end, RAJA provides policies that allow developers
to have some control over the optimizations that compilers are allow to
apply during code compilation.

 ====================================== ============= ==========================
 Sequential/SIMD Execution Policies     Works with    Brief description
 ====================================== ============= ==========================
 seq_exec                               forall,       Strictly sequential
                                        kernel (For), execution.
                                        scan,
                                        sort
 simd_exec                              forall,       Try to force generation of
                                        kernel (For), SIMD instructions via
                                        scan          compiler hints in RAJA's
                                                      internal implementation.
 loop_exec                              forall,       Allow the compiler to 
                                        kernel (For), generate any optimizations
                                        scan,         that its heuristics deem
                                        sort          beneficial according;
                                                      i.e., no loop decorations
                                                      (pragmas or intrinsics) in
                                                      RAJA implementation.
 ====================================== ============= ==========================


OpenMP Parallel CPU Policies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the OpenMP CPU multithreading back-end, RAJA has policies that can be used
by themselves to execute kernels. In particular, they create an OpenMP parallel
region and execute a kernel within it. To distinguish these in this discussion,
we refer to these as **full policies**. These policies are provided 
to users for convenience in common use cases. 

RAJA also provides other OpenMP policies, which we refer to as 
**partial policies**, since they need to be used in combination with other 
policies. Typically, they work by providing an *outer policy* and an 
*inner policy* as a template parameter to the outer policy. These give users 
flexibility to create more complex execution patterns.


.. note:: To control the number of threads used by OpenMP policies,
          set the value of the environment variable 'OMP_NUM_THREADS' (which is
          fixed for duration of run), or call the OpenMP routine
          'omp_set_num_threads(nthreads)' in your application, which allows 
          one to change the number of threads at run time.

The full policies are described in the following table. Partial policies
are described in other tables below.

 ========================================= ============= =======================
 OpenMP CPU Full Policies                  Works with    Brief description
 ========================================= ============= =======================
 omp_parallel_for_exec                     forall,       Same as applying 
                                           kernel (For), 'omp parallel for' 
                                           scan,         pragma
                                           sort
 omp_parallel_for_static_exec<ChunkSize>   forall,       Same as applying
                                           kernel (For)  'omp parallel for
                                                         schedule(static,
                                                         ChunkSize)'
 omp_parallel_for_dynamic_exec<ChunkSize>  forall,       Same as applying
                                           kernel (For)  'omp parallel for
                                                         schedule(dynamic,
                                                         ChunkSize)'
 omp_parallel_for_guided_exec<ChunkSize>   forall,       Same as applying
                                           kernel (For)  'omp parallel for
                                                         schedule(guided,
                                                         ChunkSize)'
 omp_parallel_for_runtime_exec             forall,       Same as applying
                                           kernel (For)  'omp parallel for
                                                         schedule(runtime)'
 ========================================= ============= =======================

.. note:: For the OpenMP scheduling policies above that take a ``ChunkSize``
          parameter, the chunk size is optional. If not provided, the 
          default chunk size that OpenMP applies will be used, which may
          be specific to the OpenMP implementation in use. For this case,
          the RAJA policy syntax is 
          ``omp_parallel_for_{static|dynamic|guided}_exec< >``, which will 
          result in the OpenMP pragma 
          ``omp parallel for schedule({static|dynamic|guided})`` being applied. 

RAJA provides an (outer) OpenMP CPU policy to create a parallel region in 
which to execute a kernel. It requires an inner policy that defines how a 
kernel will execute in parallel inside the region.

 ====================================== ============= ==========================
 OpenMP CPU Outer Policies              Works with    Brief description
 ====================================== ============= ==========================
 omp_parallel_exec<InnerPolicy>         forall,       Creates OpenMP parallel
                                        kernel (For), region and requires an
                                        scan          **InnerPolicy**. Same as
                                                      applying 'omp parallel'
                                                      pragma.
 ====================================== ============= ==========================

Finally, we summarize the inner policies that RAJA provides for OpenMP.
These policies are passed to the RAJA ``omp_parallel_exec`` outer policy as 
a template argument as described above.

 ====================================== ============= ==========================
 OpenMP CPU Inner Policies              Works with    Brief description
 ====================================== ============= ==========================
 omp_for_exec                           forall,       Parallel execution within
                                        kernel (For), *existing parallel 
                                        scan          region*; i.e., 
                                                      apply 'omp for' pragma. 
 omp_for_static_exec<ChunkSize>         forall,       Same as applying
                                        kernel (For)  'omp for
                                                      schedule(static,
                                                      ChunkSize)'
 omp_for_nowait_static_exec<ChunkSize>  forall,       Same as applying
                                        kernel (For)  'omp for
                                                      schedule(static,
                                                      ChunkSize) nowait'
 omp_for_dynamic_exec<ChunkSize>        forall,       Same as applying
                                        kernel (For)  'omp for
                                                      schedule(dynamic,
                                                      ChunkSize)'
 omp_for_guided_exec<ChunkSize>         forall,       Same as applying
                                        kernel (For)  'omp for
                                                      schedule(guided,
                                                      ChunkSize)'
 omp_for_runtime_exec                   forall,       Same as applying
                                        kernel (For)  'omp for
                                                      schedule(runtime)'
 omp_parallel_collapse_exec             kernel        Use in Collapse statement
                                        (Collapse +   to parallelize multiple
                                        ArgList)      loop levels in loop nest
                                                      indicated using ArgList
 ====================================== ============= ==========================

.. important:: **RAJA only provides a nowait policy option for static 
               scheduling** since that is the only schedule case that can be 
               used with nowait and be correct in general when executing 
               multiple loops in a single parallel region. Paraphrasing the 
               OpenMP standard:
               *programs that depend on which thread executes a particular
               loop iteration under any circumstance other than static schedule
               are non-conforming.*

.. note:: As in the RAJA full policies for OpenMP scheduling, the ``ChunkSize``
          is optional. If not provided, the default chunk size that the OpenMP 
          implementation applies will be used.

.. note:: As noted above, RAJA inner OpenMP policies must only be used within an
          **existing** parallel region to work properly. Embedding an inner 
          policy inside the RAJA outer ``omp_parallel_exec`` will allow you to 
          apply the OpenMP execution prescription specified by the policies to 
          a single kernel. To support use cases with multiple kernels inside an
          OpenMP parallel region, RAJA provides a **region** construct that 
          takes a template argument to specify the execution back-end. For 
          example::

            RAJA::region<RAJA::omp_parallel_region>([=]() {

              RAJA::forall<RAJA::omp_for_nowait_static_exec< > >(segment, 
                [=] (int idx) {
                  // do something at iterate 'idx'
                }
              );

              RAJA::forall<RAJA::omp_for_static_exec< > >(segment, 
                [=] (int idx) {
                  // do something else at iterate 'idx'
                }
              );

            });

          Here, the ``RAJA::region<RAJA::omp_parallel_region>`` method call
          creates an OpenMP parallel region, which contains two ``RAJA::forall``
          kernels. The first uses the ``RAJA::omp_for_nowait_static_exec< >`` 
          policy, meaning that no thread synchronization is needed after the 
          kernel. Thus, threads can start working on the second kernel while 
          others are still working on the first kernel. I general, this will
          be correct when the segments used in the two kernels are the same,
          each loop is data parallel, and static scheduling is applied to both
          loops. The second kernel uses the ``RAJA::omp_for_static_exec`` 
          policy, which means that all threads will complete before the kernel 
          exits. In this example, this is not really needed since there is no 
          more code to execute in the parallel region and there is an implicit 
          barrier at the end of it.

Threading Building Block (TBB) Parallel CPU Policies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RAJA provides a basic set of TBB execution policies for use with the 
RAJA TBB back-end, which supports a subset of RAJA features.

 ====================================== ============= ==========================
 Threading Building Blocks Policies     Works with    Brief description
 ====================================== ============= ==========================
 tbb_for_exec                           forall,       Execute loop iterations.
                                        kernel (For), as tasks in parallel using
                                        scan          TBB ``parallel_for``
                                                      method.
 tbb_for_static<CHUNK_SIZE>             forall,       Same as above, but use.
                                        kernel (For), a static scheduler with
                                        scan          given chunk size.
 tbb_for_dynamic                        forall,       Same as above, but use
                                        kernel (For), a dynamic scheduler.
                                        scan
 ====================================== ============= ==========================

.. note:: To control the number of TBB worker threads used by these policies:
          set the value of the environment variable 'TBB_NUM_WORKERS' (which is
          fixed for duration of run), or create a 'task_scheduler_init' object::

            tbb::task_scheduler_init TBBinit( nworkers );

            // do some parallel work

            TBBinit.terminate();
            TBBinit.initialize( new_nworkers );

            // do some more parallel work

          This allows changing number of workers at run time.


GPU Policies for CUDA and HIP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RAJA policies for GPU execution using CUDA or HIP are essentially identical. 
The only difference is that CUDA policies have the prefix ``cuda_`` and HIP 
policies have the prefix ``hip_``.

 ======================================== ============= ========================
 CUDA/HIP Execution Policies              Works with    Brief description
 ======================================== ============= ========================
 cuda/hip_exec<BLOCK_SIZE>                forall,       Execute loop iterations
                                          scan,         in a GPU kernel launched
                                          sort          with given thread-block
                                                        size. Note that the 
                                                        thread-block size must
                                                        be provided, there is
                                                        no default provided.
 cuda/hip_thread_x_direct                 kernel (For)  Map loop iterates
                                                        directly to GPU threads
                                                        in x-dimension, one
                                                        iterate per thread
                                                        (see note below about
                                                        limitations)
 cuda/hip_thread_y_direct                 kernel (For)  Same as above, but map
                                                        to threads in y-dim
 cuda/hip_thread_z_direct                 kernel (For)  Same as above, but map
                                                        to threads in z-dim
 cuda/hip_thread_x_loop                   kernel (For)  Similar to 
                                                        thread-x-direct
                                                        policy, but use a
                                                        block-stride loop which
                                                        doesn't limit number of
                                                        loop iterates
 cuda/hip_thread_y_loop                   kernel (For)  Same as above, but for
                                                        threads in y-dimension
 cuda/hip_thread_z_loop                   kernel (For)  Same as above, but for
                                                        threads in z-dimension
 cuda/hip_flatten_block_threads_{xyz}     Launch (Loop) Reshapes threads in a
                                                        multi-dimensional thread
                                                        team into one-dimension,
                                                        accepts any permutation
                                                        of dimensions
                                                        (expt namespace)
 cuda/hip_block_x_direct                  kernel (For)  Map loop iterates
                                                        directly to GPU thread
                                                        blocks in x-dimension,
                                                        one iterate per block
 cuda/hip_block_y_direct                  kernel (For)  Same as above, but map
                                                        to blocks in y-dimension
 cuda/hip_block_z_direct                  kernel (For)  Same as above, but map
                                                        to blocks in z-dimension
 cuda/hip_block_x_loop                    kernel (For)  Similar to 
                                                        block-x-direct policy, 
                                                        but use a grid-stride 
                                                        loop.
 cuda/hip_block_y_loop                    kernel (For)  Same as above, but use
                                                        blocks in y-dimension
 cuda/hip_block_z_loop                    kernel (For)  Same as above, but use
                                                        blocks in z-dimension
 cuda/hip_global_thread_x                 Launch (Loop) Creates a unique thread
                                                        id for each thread on 
                                                        x-dimension of the grid
                                                        (expt namespace)
 cuda/hip_global_thread_y                 Launch (Loop) Same as above, but uses
                                                        threads in y-dimension
                                                        (expt namespace)
 cuda/hip_global_thread_z                 Launch (Loop) Same as above, but uses
                                                        threads in z-dimension
                                                        (expt namespace)
 cuda/hip_warp_direct                     kernel (For)  Map work to threads
                                                        in a warp directly.
                                                        Cannot be used in
                                                        conjunction with
                                                        cuda/hip_thread_x_* 
                                                        policies.
                                                        Multiple warps can be
                                                        created by using
                                                        cuda/hip_thread_y/z_*
                                                        policies.
 cuda/hip_warp_loop                       kernel (For)  Policy to map work to
                                                        threads in a warp using
                                                        a warp-stride loop.
                                                        Cannot be used in
                                                        conjunction with
                                                        cuda/hip_thread_x_* 
                                                        policies.
                                                        Multiple warps can be
                                                        created by using
                                                        cuda/hip_thread_y/z_*
                                                        policies.
 cuda/hip_warp_masked_direct<BitMask<..>> kernel (For)  Policy to map work
                                                        directly to threads in a
                                                        warp using a bit mask.
                                                        Cannot be used in
                                                        conjunction with
                                                        cuda/hip_thread_x_* 
                                                        policies.
                                                        Multiple warps can
                                                        be created by using
                                                        cuda/hip_thread_y/z_*
                                                        policies.
 cuda/hip_warp_masked_loop<BitMask<..>>   kernel (For)  Policy to map work to
                                                        threads in a warp using
                                                        a bit mask and a 
                                                        warp-stride loop. Cannot
                                                        be used in conjunction 
                                                        with cuda/hip_thread_x_*
                                                        policies. Multiple warps                                                        can be created by using
                                                        cuda/hip_thread_y/z_*
                                                        policies.
 cuda/hip_block_reduce                    kernel        Perform a reduction
                                          (Reduce)      across a single GPU
                                                        thread block.
 cuda/_warp_reduce                        kernel        Perform a reduction
                                          (Reduce)      across a single GPU
                                                        thread warp.
 ======================================== ============= ========================

Several notable constraints apply to RAJA CUDA/HIP *thread-direct* policies.

.. note:: * Repeating thread direct policies with the same thread dimension
            in perfectly nested loops is not recommended. Your code may do
            something, but likely will not do what you expect and/or be correct.
          * If multiple thread direct policies are used in a kernel (using
            different thread dimensions), the product of sizes of the
            corresponding iteration spaces cannot be greater than the
            maximum allowable threads per block. Typically, this is
            1024 threads per block. Attempting to execute a kernel with more
            than the maximum allowed the CUDA runtime
            to complain about *illegal launch parameters.*
          * **Thread-direct policies are recommended only for certain loop
            patterns, such as tiling.**

Several notes regarding CUDA/HIP thread and block *loop* policies are also
good to know.

.. note:: * There is no constraint on the product of sizes of the associated
            loop iteration space.
          * These polices allow having a larger number of iterates than
            threads in the x, y, or z thread dimension.
          * **CUDA/HIP thread and block loop policies are recommended for most
            loop patterns.**

Finally

.. note:: CUDA/HIP block-direct policies may be preferable to block-loop
          policies in situations where block load balancing may be an issue
          as the block-direct policies may yield better performance.


GPU Policies for SYCL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 ======================================== ============= ========================
 SYCL Execution Policies                  Works with    Brief description
 ======================================== ============= ========================
 sycl_exec<WORK_GROUP_SIZE>               forall,       Execute loop iterations
                                                        in a GPU kernel launched
                                                        with given work group
                                                        size.
 sycl_global_0<WORK_GROUP_SIZE>           kernel (For)  Map loop iterates
                                                        directly to GPU global
                                                        ids in first
                                                        dimension, one iterate 
                                                        per work item. Group
                                                        execution into work
                                                        groups of given size. 
 sycl_global_1<WORK_GROUP_SIZE>           kernel (For)  Same as above, but map
                                                        to global ids in second
                                                        dim
 sycl_global_2<WORK_GROUP_SIZE>           kernel (For)  Same as above, but map
                                                        to global ids in third 
                                                        dim
 sycl_local_0_direct                      kernel (For)  Map loop iterates
                                                        directly to GPU work
                                                        items in first
                                                        dimension, one iterate 
                                                        per work item (see note 
                                                        below about limitations)
 sycl_local_1_direct                      kernel (For)  Same as above, but map
                                                        to work items in second
                                                        dim
 sycl_local_2_direct                      kernel (For)  Same as above, but map
                                                        to work items in third 
                                                        dim
 sycl_local_0_loop                        kernel (For)  Similar to 
                                                        local-1-direct policy, 
                                                        but use a work 
                                                        group-stride loop which
                                                        doesn't limit number of
                                                        loop iterates
 sycl_local_1_loop                        kernel (For)  Same as above, but for
                                                        work items in second 
                                                        dimension
 sycl_local_2_loop                        kernel (For)  Same as above, but for
                                                        work items in third 
                                                        dimension
 sycl_group_0_direct                      kernel (For)  Map loop iterates
                                                        directly to GPU group
                                                        ids in first dimension, 
                                                        one iterate per group
 sycl_group_1_direct                      kernel (For)  Same as above, but map
                                                        to groups in second 
                                                        dimension
 sycl_group_2_direct                      kernel (For)  Same as above, but map
                                                        to groups in third 
                                                        dimension
 sycl_group_0_loop                        kernel (For)  Similar to 
                                                        group-1-direct policy, 
                                                        but use a group-stride 
                                                        loop.
 sycl_group_1_loop                        kernel (For)  Same as above, but use
                                                        groups in second 
                                                        dimension
 sycl_group_2_loop                        kernel (For)  Same as above, but use
                                                        groups in third 
                                                        dimension

 ======================================== ============= ========================

OpenMP Target Offload Policies 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RAJA provides policies to use OpenMP to offload kernel execution to a GPU 
device, for example. They are summarized in the following table.

 ====================================== ============= ==========================
 OpenMP Target Execution Policies       Works with    Brief description
 ====================================== ============= ==========================
 omp_target_parallel_for_exec<#>        forall,       Create parallel target
                                        kernel(For)   region and execute with
                                                      given number of threads
                                                      per team inside it. Number
                                                      of teams is calculated
                                                      internally; i.e.,
                                                      apply ``omp teams
                                                      distribute parallel for
                                                      num_teams(iteration space
                                                      size/#)
                                                      thread_limit(#)`` pragma
 omp_target_parallel_collapse_exec      kernel        Similar to above, but
                                        (Collapse)    collapse
                                                      *perfectly-nested*
                                                      loops, indicated in
                                                      arguments to RAJA
                                                      Collapse statement. Note:
                                                      compiler determines number
                                                      of thread teams and
                                                      threads per team
 ====================================== ============= ==========================

.. _indexsetpolicy-label:

-----------------------------------------------------
RAJA IndexSet Execution Policies
-----------------------------------------------------

When an IndexSet iteration space is used in RAJA by passing an IndexSet
to a ``RAJA::forall`` method, for example, an index set execution policy is 
required. An index set execution policy is a **two-level policy**: an 'outer' 
policy for iterating over segments in the index set, and an 'inner' policy 
used to execute the iterations defined by each segment. An index set execution 
policy type has the form::

  RAJA::ExecPolicy< segment_iteration_policy, segment_execution_policy >

In general, any policy that can be used with a ``RAJA::forall`` method
can be used as the segment execution policy. The following policies are
available to use for the outer segment iteration policy:

====================================== =========================================
Execution Policy                       Brief description
====================================== =========================================
**Serial**
seq_segit                              Iterate over index set segments
                                       sequentially.

**OpenMP CPU multithreading**
omp_parallel_segit                     Create OpenMP parallel region and
                                       iterate over segments in parallel inside                                        it; i.e., apply ``omp parallel for``
                                       pragma on loop over segments.
omp_parallel_for_segit                 Same as above.

**Intel Threading Building Blocks**
tbb_segit                              Iterate over index set segments in
                                       parallel using a TBB 'parallel_for'
                                       method.
====================================== =========================================

-------------------------
Parallel Region Policies
-------------------------

Earlier, we discussed using the ``RAJA::region`` construct to
execute multiple kernels in an OpenMP parallel region. To support source code 
portability, RAJA provides a sequential region concept that can be used to 
surround code that uses execution back-ends other than OpenMP. For example::

  RAJA::region<RAJA::seq_region>([=]() {

     RAJA::forall<RAJA::loop_exec>(segment, [=] (int idx) {
         // do something at iterate 'idx'
     } );

     RAJA::forall<RAJA::loop_exec>(segment, [=] (int idx) {
         // do something else at iterate 'idx'
     } );

   });

.. note:: The sequential region specialization is essentially a *pass through*
          operation. It is provided so that if you want to turn off OpenMP in
          your code, for example, you can simply replace the region policy 
          type and you do not have to change your algorithm source code.


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

======================= ============= ==========================================
Reduction Policy        Loop Policies Brief description
                        to Use With
======================= ============= ==========================================
seq_reduce              seq_exec,     Non-parallel (sequential) reduction.
                        loop_exec
omp_reduce              any OpenMP    OpenMP parallel reduction.
                        policy
omp_reduce_ordered      any OpenMP    OpenMP parallel reduction with result
                        policy        guaranteed to be reproducible.
omp_target_reduce       any OpenMP    OpenMP parallel target offload reduction.
                        target policy
tbb_reduce              any TBB       TBB parallel reduction.
                        policy
cuda/hip_reduce         any CUDA/HIP  Parallel reduction in a CUDA/HIP kernel
                        policy        (device synchronization will occur when
                                      reduction value is finalized).
cuda/hip_reduce_atomic  any CUDA/HIP  Same as above, but reduction may use CUDA
                        policy        atomic operations.
sycl_reduce             any SYCL      Reduction in a SYCL kernel (device 
                        policy        synchronization will occur when the 
                                      reduction value is finalized).
======================= ============= ==========================================

.. note:: RAJA reductions used with SIMD execution policies are not
          guaranteed to generate correct results. So they should not be used
          for kernels containing reductions.

.. _atomicpolicy-label:

-------------------------
Atomic Policies
-------------------------

Each RAJA atomic operation must be defined with an 'atomic policy'
type. Atomic policy types are distinct from loop execution policy types.

.. note :: An atomic policy type must be consistent with the loop execution
           policy for the kernel in which the atomic operation is used. The
           following table summarizes RAJA atomic policies and usage.

============================= ============= ========================================
Atomic Policy                 Loop Policies Brief description
                              to Use With
============================= ============= ========================================
seq_atomic                    seq_exec,     Atomic operation performed in a
                              loop_exec     non-parallel (sequential) kernel.
omp_atomic                    any OpenMP    Atomic operation performed in an OpenMP.
                              policy        multithreading or target kernel; i.e.,
                                              apply ``omp atomic`` pragma.
cuda/hip/sycl_atomic          any           Atomic operation performed in a
                              CUDA/HIP/SYCL CUDA/HIP/SYCL kernel.
                              policy        

cuda/hip_atomic_explicit      any CUDA/HIP  Atomic operation performed in a CUDA/HIP
                              policy        kernel that may also be used in a host
                                            execution context. The atomic policy
                                            takes a host atomic policy template
                                            argument. See additional explanation 
                                            and example below.
builtin_atomic                seq_exec,     Compiler *builtin* atomic operation.
                              loop_exec,
                              any OpenMP
                              policy
auto_atomic                   seq_exec,     Atomic operation *compatible* with loop
                              loop_exec,    execution policy. See example below.
                              any OpenMP    Can not be used inside cuda/hip
                              policy,       explicit atomic policies.
                              any 
                              CUDA/HIP/SYCL
                              policy
============================= ============= ========================================

.. note:: The ``cuda_atomic_explicit`` and ``hip_atomic_explicit`` policies
          take a host atomic policy template parameter. They are intended to
          be used with kernels that are host-device decorated to be used in
          either a host or device execution context.

Here is an example illustrating use of the ``cuda_atomic_explicit`` policy::

  auto kernel = [=] RAJA_HOST_DEVICE (RAJA::Index_type i) {
    RAJA::atomicAdd< RAJA::cuda_atomic_explicit<omp_atomic> >(&sum, 1);
  };

  RAJA::forall< RAJA::cuda_exec<BLOCK_SIZE> >(RAJA::TypedRangeSegment<int> seg(0, N), kernel);

  RAJA::forall< RAJA::omp_parallel_for_exec >(RAJA::TypedRangeSegment<int> seg(0, N), kernel);

In this case, the atomic operation knows when it is compiled for the device
in a CUDA kernel context and the CUDA atomic operation is applied. Similarly
when it is compiled for the host in an OpenMP kernel the omp_atomic policy is
used and the OpenMP version of the atomic operation is applied.

Here is an example illustrating use of the ``auto_atomic`` policy::

  RAJA::forall< RAJA::cuda_exec<BLOCK_SIZE> >(RAJA::TypedRangeSegment<int> seg(0, N),
    [=] RAJA_DEVICE (RAJA::Index_type i) {

    RAJA::atomicAdd< RAJA::auto_atomic >(&sum, 1);

  });

In this case, the atomic operation knows that it is used in a CUDA kernel
context and the CUDA atomic operation is applied. Similarly, if an OpenMP
execution policy was used, the OpenMP version of the atomic operation would
be used.

.. note:: * There are no RAJA atomic policies for TBB (Intel Threading Building
            Blocks) execution contexts since reductions are not supported
            for the RAJA TBB back-end.
          * The ``builtin_atomic`` policy may be preferable to the
            ``omp_atomic`` policy in terms of performance.

.. _localarraypolicy-label:

----------------------------
Local Array Memory Policies
----------------------------

``RAJA::LocalArray`` types must use a memory policy indicating
where the memory for the local array will live. These policies are described
in :ref:`feat-local_array-label`.

The following memory policies are available to specify memory allocation
for ``RAJA::LocalArray`` objects:

  *  ``RAJA::cpu_tile_mem`` - Allocate CPU memory on the stack
  *  ``RAJA::cuda/hip_shared_mem`` - Allocate CUDA or HIP shared memory
  *  ``RAJA::cuda/hip_thread_mem`` - Allocate CUDA or HIP thread private memory


.. _loop_elements-kernelpol-label:

--------------------------------
RAJA Kernel Execution Policies
--------------------------------

RAJA kernel execution policy constructs form a simple domain specific language
for composing and transforming complex loops that relies
**solely on standard C++14 template support**.
RAJA kernel policies are constructed using a combination of *Statements* and
*Statement Lists*. A RAJA Statement is an action, such as execute a loop,
invoke a lambda, set a thread barrier, etc. A StatementList is an ordered list
of Statements that are composed in the order that they appear in the kernel
policy to construct a kernel. A Statement may contain an enclosed StatmentList. Thus, a ``RAJA::KernelPolicy`` type is really just a StatementList.

The main Statement types provided by RAJA are ``RAJA::statement::For`` and
``RAJA::statement::Lambda``, that we discussed in 
:ref:`loop_elements-kernel-label`. 
A ``RAJA::statement::For<ArgID, ExecPolicy, Enclosed Satements>`` type 
indicates a for-loop structure. The ``ArgID`` parameter is an integral constant
that identifies the position of the iteration space in the iteration space 
tuple passed to the ``RAJA::kernel`` method to be used for the loop. The 
``ExecPolicy`` is the RAJA execution policy to use on the loop, which is 
similar to ``RAJA::forall`` usage. The ``EnclosedStatements`` type is a 
nested template parameter that contains whatever is needed to execute the 
kernel and which forms a valid StatementList. The 
``RAJA::statement::Lambda<LambdaID>``
type invokes the lambda expression corresponding to its position 'LambdaID' 
in the sequence of lambda expressions in the ``RAJA::kernel`` argument list. 
For example, a simple sequential for-loop::

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
    RAJA::make_tuple(range),
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
the ``RAJA::kernel`` examples in :ref:`tutorial-label`.

.. note:: All of the statement types described below are in the namespace 
          ``RAJA::statement``. For brevity, we omit the namespaces in
          the discussion in this section.

.. note::  ``RAJA::kernel_param`` functions similarly to ``RAJA::kernel`` 
           except that the second argument is a *tuple of parameters* used 
           in a kernel for local arrays, thread local variables, tiling 
           information, etc.

Several RAJA statements can be specialized with auxilliary types, which are
described in :ref:`auxilliarypolicy_label`.

The following list contains the most commonly used statement types.

* ``For< ArgId, ExecPolicy, EnclosedStatements >`` abstracts a for-loop associated with kernel iteration space at tuple index ``ArgId``, to be run with ``ExecPolicy`` execution policy, and containing the ``EnclosedStatements`` which are executed for each loop iteration.

* ``Lambda< LambdaId >`` invokes the lambda expression that appears at position 'LambdaId' in the sequence of lambda arguments. With this statement, the lambda expression must accept all arguments associated with the tuple of iteration space segments and tuple of parameters (if kernel_param is used).

* ``Lambda< LambdaId, Args...>`` extends the Lambda statement. The second template parameter indicates which arguments (e.g., which segment iteration variables) are passed to the lambda expression.

* ``Collapse< ExecPolicy, ArgList<...>, EnclosedStatements >`` collapses multiple perfectly nested loops specified by tuple iteration space indices in ``ArgList``, using the ``ExecPolicy`` execution policy, and places ``EnclosedStatements`` inside the collapsed loops which are executed for each iteration. **Note that this only works for CPU execution policies (e.g., sequential, OpenMP).** It may be available for CUDA in the future if such use cases arise.

There is one statement specific to OpenMP kernels. 

* ``OmpSyncThreads`` applies the OpenMP ``#pragma omp barrier`` directive.

Statement types that launch CUDA or HIP GPU kernels are listed next. They work 
similarly for each back-end and their names are distinguished by the prefix 
``Cuda`` or ``Hip``. For example, ``CudaKernel`` or ``HipKernel``.

* ``Cuda/HipKernel< EnclosedStatements>`` launches ``EnclosedStatements`` as a GPU kernel; e.g., a loop nest where the iteration spaces of each loop level are associated with threads and/or thread blocks as described by the execution policies applied to them. This kernel launch is synchronous.

* ``Cuda/HipKernelAsync< EnclosedStatements>`` asynchronous version of Cuda/HipKernel.

* ``Cuda/HipKernelFixed<num_threads, EnclosedStatements>`` similar to Cuda/HipKernel but enables a fixed number of threads (specified by num_threads). This kernel launch is synchronous.

* ``Cuda/HipKernelFixedAsync<num_threads, EnclosedStatements>`` asynchronous version of Cuda/HipKernelFixed.

* ``CudaKernelFixedSM<num_threads, min_blocks_per_sm, EnclosedStatements>`` similar to CudaKernelFixed but enables a minimum number of blocks per sm (specified by min_blocks_per_sm), this can help increase occupancy. This kernel launch is synchronous.  **Note: there is no HIP variant of this statement.**

* ``CudaKernelFixedSMAsync<num_threads, min_blocks_per_sm, EnclosedStatements>`` asynchronous version of CudaKernelFixedSM. **Note: there is no HIP variant of this statement.**

* ``Cuda/HipKernelOcc<EnclosedStatements>`` similar to CudaKernel but uses the CUDA occupancy calculator to determine the optimal number of threads/blocks. Statement is intended for use with RAJA::cuda/hip_block_{xyz}_loop policies. This kernel launch is synchronous.

* ``Cuda/HipKernelOccAsync<EnclosedStatements>`` asynchronous version of Cuda/HipKernelOcc.

* ``Cuda/HipKernelExp<num_blocks, num_threads, EnclosedStatements>`` similar to CudaKernelOcc but with the flexibility to fix the number of threads and/or blocks and let the CUDA occupancy calculator determine the unspecified values. This kernel launch is synchronous.

* ``Cuda/HipKernelExpAsync<num_blocks, num_threads, EnclosedStatements>`` asynchronous version of Cuda/HipKernelExp.

* ``Cuda/HipSyncThreads`` invokes CUDA or HIP ``__syncthreads()`` barrier.

* ``Cuda/HipSyncWarp`` invokes CUDA ``__syncwarp()`` barrier. Warp sync is not supported in HIP, so the HIP variant is a no-op.

Statement types that launch SYCL kernels are listed next. 

* ``SyclKernel<EnclosedStatements>`` launches ``EnclosedStatements`` as a SYCL kernel.  This kernel launch is synchronous.

* ``SyclKernelAsync<EnclosedStatements>`` asynchronous version of SyclKernel.

RAJA provides statements to define loop tiling which can improve performance; 
e.g., by allowing CPU cache blocking or use of GPU shared memory. 

* ``Tile< ArgId, TilePolicy, ExecPolicy, EnclosedStatements >`` abstracts an outer tiling loop containing an inner for-loop over each tile. The ``ArgId`` indicates which entry in the iteration space tuple to which the tiling loop applies and the ``TilePolicy`` specifies the tiling pattern to use, including its dimension. The ``ExecPolicy`` and ``EnclosedStatements`` are similar to what they represent in a ``statement::For`` type.

* ``TileTCount< ArgId, ParamId, TilePolicy, ExecPolicy, EnclosedStatements >`` abstracts an outer tiling loop containing an inner for-loop over each tile, **where it is necessary to obtain the tile number in each tile**. The ``ArgId`` indicates which entry in the iteration space tuple to which the loop applies and the ``ParamId`` indicates the position of the tile number in the parameter tuple. The ``TilePolicy`` specifies the tiling pattern to use, including its dimension. The ``ExecPolicy`` and ``EnclosedStatements`` are similar to what they represent in a ``statement::For`` type.

* ``ForICount< ArgId, ParamId, ExecPolicy, EnclosedStatements >`` abstracts an inner for-loop within an outer tiling loop **where it is necessary to obtain the local iteration index in each tile**. The ``ArgId`` indicates which entry in the iteration space tuple to which the loop applies and the ``ParamId`` indicates the position of the tile index parameter in the parameter tuple. The ``ExecPolicy`` and ``EnclosedStatements`` are similar to what they represent in a ``statement::For`` type.

It is often advantageous to use local arrays for data accessed in tiled loops.
RAJA provides a statement for allocating data in a :ref:`feat-local_array-label`
object according to a memory policy. See :ref:`localarraypolicy-label` for more information about such policies.

* ``InitLocalMem< MemPolicy, ParamList<...>, EnclosedStatements >`` allocates memory for a ``RAJA::LocalArray`` object used in kernel. The ``ParamList`` entries indicate which local array objects in a tuple will be initialized. The ``EnclosedStatements`` contain the code in which the local array will be accessed; e.g., initialization operations.

RAJA provides some statement types that apply in specific kernel scenarios.

* ``Reduce< ReducePolicy, Operator, ParamId, EnclosedStatements >`` reduces a value across threads in a multithreaded code region to a single thread. The ``ReducePolicy`` is similar to what it represents for RAJA reduction types. ``ParamId`` specifies the position of the reduction value in the parameter tuple passed to the ``RAJA::kernel_param`` method. ``Operator`` is the binary operator used in the reduction; typically, this will be one of the operators that can be used with RAJA scans (see :ref:`feat-scanops-label`). After the reduction is complete, the ``EnclosedStatements`` execute on the thread that received the final reduced value.

* ``If< Conditional >`` chooses which portions of a policy to run based on run-time evaluation of conditional statement; e.g., true or false, equal to some value, etc.

* ``Hyperplane< ArgId, HpExecPolicy, ArgList<...>, ExecPolicy, EnclosedStatements >`` provides a hyperplane (or wavefront) iteration pattern over multiple indices. A hyperplane is a set of multi-dimensional index values: i0, i1, ... such that h = i0 + i1 + ... for a given h. Here, ``ArgId`` is the position of the loop argument we will iterate on (defines the order of hyperplanes), ``HpExecPolicy`` is the execution policy used to iterate over the iteration space specified by ArgId (often sequential), ``ArgList`` is a list of other indices that along with ArgId define a hyperplane, and ``ExecPolicy`` is the execution policy that applies to the loops in ``ArgList``. Then, for each iteration, everything in the ``EnclosedStatements`` is executed.


.. _auxilliarypolicy_label:

Auxilliary Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following list summarizes auxilliary types used in the above statements. These
types live in the ``RAJA`` namespace.

  * ``tile_fixed<TileSize>`` tile policy argument to a ``Tile`` or ``TileTCount`` statement; partitions loop iterations into tiles of a fixed size specified by ``TileSize``. This statement type can be used as the ``TilePolicy`` template parameter in the ``Tile`` statements above.
 
  * ``tile_dynamic<ParamIdx>`` TilePolicy argument to a Tile or TileTCount statement; partitions loop iterations into tiles of a size specified by a ``TileSize{}`` positional parameter argument. This statement type can be used as the ``TilePolicy`` template paramter in the ``Tile`` statements above.

  * ``Segs<...>`` argument to a Lambda statement; used to specify which segments in a tuple will be used as lambda arguments.

  * ``Offsets<...>`` argument to a Lambda statement; used to specify which segment offsets in a tuple will be used as lambda arguments.

  * ``Params<...>`` argument to a Lambda statement; used to specify which params in a tuple will be used as lambda arguments.

  * ``ValuesT<T, ...>`` argument to a Lambda statement; used to specify compile time constants, of type T, that will be used as lambda arguments.

Examples that show how to use a variety of these statement types can be found
in :ref:`loop_elements-kernel-label`.
