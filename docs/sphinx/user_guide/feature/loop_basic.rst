.. ##
.. ## Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
.. ## and other RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _loop_elements-label:

==============================================
Elements of Loop Execution
==============================================

In this section, we describe the basic elements of RAJA loop kernel 
execution.  ``RAJA::forall``, ``RAJA::kernel``, and ``RAJA::expt::launch`` 
template methods comprise the RAJA interface for loop 
execution. ``RAJA::forall`` methods execute simple, non-nested loops, 
``RAJA::kernel`` methods support nested loops and other complex loop 
kernels and transformations, and ``RAJA::expt::launch`` creates an execution 
space in which algorithms are expressed in terms of nested loops using 
the ``RAJA::expt::loop`` method.

.. note:: * The ``forall`` , and ``kernel`` methods are in the
            namespace ``RAJA``, while ``launch`` is found under
            the RAJA namespace for experimental features ``RAJA::expt``.

          * A ``RAJA::forall`` loop execution method is a template on an
            *execution policy* type. A ``RAJA::forall`` method takes two 
            arguments:

              * an iteration space object, such as a contiguous range of loop
                indices, and
              * a single lambda expression representing the loop body.

          * Each ``RAJA::kernel`` method is a template on a policy that
            contains statements with *execution policy* types appropriate for
            the kernel structure; e.g., an execution policy for each level in a
            loop nest. A ``RAJA::kernel`` method takes multiple arguments:

              * a *tuple* of iteration space objects, and
              * one or more lambda expressions representing portions of
                the loop kernel body.

          * The ``RAJA::expt::launch`` method is a template on both host and
            device policies to create an execution space for kernels.
            Since both host and device poilices are specified, the launch 
            method can be used to select at run-time whether to run a kernel
            on the host or device.  Algorithms are expressed inside the 
            execution space as nested loops using ``RAJA::loop`` methods.

              * Hierarchical parallelism can be expressed using the thread and
                thread-team model with ``RAJA::expt::loop`` methods as found in
                programming models such as CUDA/HIP.

Various examples showing how to use ``RAJA::forall``, ``RAJA::kernel``, ``RAJA::launch``
methods may be found in the :ref:`tutorial-label`.

For more information on RAJA execution policies and iteration space constructs, 
see :ref:`policies-label` and :ref:`index-label`, respectively. 

.. _loop_elements-forall-label:

---------------------------
Simple Loops (RAJA::forall)
---------------------------

As noted earlier, a ``RAJA::forall`` template executes simple 
(i.e., non-nested) loops. For example, a C-style loop that adds two vectors,
like this::

  for (int i = 0; i < N; ++i) {
    c[i] = a[i] + b[i];
  }

may be written using RAJA as::

  RAJA::forall<exec_policy>(RAJA::RangeSegment(0, N), [=] (int i) {
    c[i] = a[i] + b[i];
  });

A ``RAJA::forall`` method is a template on an execution policy type and takes
two arguments: an object describing the loop iteration space, such as a RAJA 
range segment (shown here), and a lambda expression for the loop body. Applying 
different loop execution policies enables the loop to run in different ways; 
e.g., using different programming model back-ends. Different iteration space 
objects enable the loop iterates to be partitioned, reordered, run in 
different threads, etc. 

.. note:: Changing loop execution policy types and iteration space constructs
          enables loops to run in different ways by recompiling the code and 
          without modifying the loop kernel code.

While loop execution using ``RAJA::forall`` methods is a subset of 
``RAJA::kernel`` functionality, described next, we maintain the 
``RAJA::forall`` interface for simple loop execution because the syntax is 
simpler and less verbose for that use case.

.. note:: Data arrays in lambda expressions used with RAJA are typically 
          RAJA Views (see :ref:`view-label`) or bare pointers as shown in
          the code snippets above. Using something like 'std::vector' is
          non-portable (won't work in GPU kernels, generally) and would add 
          excessive overhead for copying data into the lambda data environment
          when captured by value.

.. _loop_elements-kernel-label:

----------------------------
Complex Loops (RAJA::kernel)
----------------------------

A ``RAJA::kernel`` template provides ways to compose and execute arbitrary 
loop nests and other complex kernels. To introduce the RAJA *kernel* interface,
consider a (N+1)-level C-style loop nest::

  for (int iN = 0; iN < NN; ++iN) {
    ...
       for (int i0 = 0; i0 < N0; ++i0) {s
         \\ inner loop body
       }
  }

Note that we could write this by nesting ``RAJA::forall`` statements and
it would work for some execution policy choices::

  RAJA::forall<exec_policyN>(IN, [=] (int iN) {
    ...
       RAJA::forall<exec_policy0>(I0, [=] (int i0)) {
         \\ inner loop body
       }
    ...
  }

However, this approach treats each loop level as an independent entity. This
makes it difficult to parallelize the levels in the loop nest together. So it
may limit the amount of parallelism that can be exposed and the types of 
parallelism that may be used. For example, if an OpenMP or CUDA
parallel execution policy is used on the outermost loop, then all inner loops
would be run sequentially in each thread. It also makes it difficult to perform 
transformations like loop interchange and loop collapse without changing the 
source code, which breaks RAJA encapsulation.

.. note:: **We do not recommend nesting ``RAJA::forall`` statements.**

The RAJA *kernel* interface facilitates parallel execution and compile-time
transformation of arbitrary loop nests and other complex loop structures. 
It can treat a complex loop structure as a single entity, which simplifies 
the ability to transform and apply different parallel execution patterns by 
changing the execution policy type and *not the kernel code*.

The loop above nest may be written using the RAJA kernel interface as::

    using KERNEL_POL = 
      RAJA::KernelPolicy< RAJA::statement::For<N, exec_policyN, 
                            ...
                              RAJA::statement::For<0, exec_policy0,
                                RAJA::statement::Lambda<0>
                              >
                            ...
                          > 
                        >;
  
    RAJA::kernel< KERNEL_POL >(
      RAJA::make_tuple(RAJA::RangeSegment(0, NN), ..., RAJA::RangeSegment(0, N0),

      [=] (int iN, ... , int i0) {
         // inner loop body
      }

    );

A ``RAJA::kernel`` method takes a ``RAJA::KernelPolicy`` type template 
parameter, and a tuple of iteration spaces and a sequence of lambda 
expressions as arguments. 

In the case we discuss here, the execution policy contains a nested sequence
of ``RAJA::statement::For`` statements, one for each level in the loop nest. 
Each ``For`` statement takes three template parameters: 

  * an integral index parameter that binds the ``For`` statement to the item 
    in the iteration space tuple corresponding to that index,
  * an execution policy type for the associated loop nest level, and
  * an *enclosed statement list* (described in :ref:`loop_elements-kernelpol-label`).

.. note:: The nesting of ``RAJA::statement::For`` types is analogous to the
          nesting of for-statements in the C-style version of the loop nest.
          One can think of the '<, >' symbols enclosing the template parameter 
          lists as being similar to the curly braces in C-style code.

Here, the innermost type in the kernel policy is a 
``RAJA::statement::Lambda<0>`` type indicating that the first lambda expression
(argument zero of the sequence of lambdas passed to the ``RAJA::kernel`` method)
will comprise the inner loop body. We only have one lambda in this example 
but, in general, we can have any number of lambdas and we can use any subset 
of them, with ``RAJA::statement::Lambda`` types placed appropriately in the
execution policy, to construct a loop kernel. For example, placing 
``RAJA::statement::Lambda`` types between ``RAJA::statement::For`` statements 
enables non-perfectly nested loops.

RAJA offers two types of lambda statements. The first as illustratated
above, requires that each lambda expression passed to a ``RAJA::kernel`` method
**must take an index argument for each iteration space in the tuple**.
With this type of lambda statement, the entire iteration space must be active 
in a containing ``For`` construct.  A compile time ``static_assert`` will be 
triggered if any of the arguments are undefined, indicating that something
is not correct.

The second type of lambda statement, an extension of the first, takes additional
template parameters which specify which iteration space indices are passed
as lambda arguments. The result is that a kernel lambda only needs to accept
iteration space index arguments that are used in the lambda body.

The kernel policy list with lambda arguments may be written as::

    using KERNEL_POL = 
      RAJA::KernelPolicy< RAJA::statement::For<N, exec_policyN, 
                            ...
                              RAJA::statement::For<0, exec_policy0,
                                RAJA::statement::Lambda<0, RAJA::Segs<N,...,0>>
                              >
                            ...
                          > 
                        >;

The template parameter ``RAJA::Segs`` is used to specify which elements in the
segment tuple are used to pass arguments to a lambda. RAJA offers other 
types such as ``RAJA::Offsets``, and ``RAJA::Params`` to identify offsets and 
parameters in segments and param tuples respectively to be used as lambda 
argumentsx. See :ref:`matrixmultiply-label` and 
:ref:`matrixtransposelocalarray-label` for detailed  examples.

.. note:: Unless lambda arguments are specified in RAJA lambda statements,
          the loop index arguments for each lambda expression used in a RAJA
          kernel loop body **must match** the contents of the 
          *iteration space tuple* in number, order, and type. Not all index 
          arguments must be used in a lambda, but they **all must appear** 
          in the lambda argument list and **all must be in active loops** to be 
          well-formed. In particular, your code will not compile if this is 
          not done correctly. If an argument is unused in a lambda expression, 
          you may include its type and omit its name in the argument list to 
          avoid compiler warnings just as one would do for a regular C++ 
          method with unused arguments.

For RAJA nested loops implemented with ``RAJA::kernel``, as shown here, the 
loop nest ordering is determined by the order of the nested policies, starting 
with the outermost loop and ending with the innermost loop. 

.. note:: The integer value that appears as the first parameter in each 
          ``RAJA::statement::For`` template indicates which iteration space 
          tuple entry or lambda index argument it corresponds to. **This 
          allows loop nesting order to be changed simply by changing the 
          ordering of the nested policy statements**. This is analogous to 
          changing the order of 'for-loop' statements in C-style nested loop 
          code.

See :ref:`matmultkernel-label` for a complete example showing RAJA nested
loop functionality and :ref:`kernelnestedreorder-label` for a detailed example 
describing nested loop reordering.

.. note:: In general, RAJA execution policies for ``RAJA::forall`` and 
          ``RAJA::kernel`` are different. A summary of all RAJA execution 
          policies that may be used with ``RAJA::forall`` or ``RAJA::kernel`` 
          may be found in :ref:`policies-label`. 

Finally, a discussion of how to construct ``RAJA::KernelPolicy`` types and 
available ``RAJA::statement`` types can be found in 
:ref:`loop_elements-kernelpol-label`.

--------------------------------
Hierachial loops (RAJA::launch)
--------------------------------

The *RAJA Launch* framework aims to unify thread/block based
programming models such as CUDA/HIP/SYCL while maintaining portability on
host backends (OpenMP, sequential). When using the ``RAJA::kernel`` 
interface, developers express all aspects of nested loop execution in the
execution policy type on which the ``RAJA::kernel`` method is templated.
In contrast, the ``RAJA::launch`` interface allows users to express 
nested loop execution in a manner that more closely reflects how one would
write conventional nested C-style for-loop code.  Additionally, *RAJA Launch* 
introduces run-time host or device selectable kernel execution. The main 
application of *RAJA Launch* is imperfectly nested loops. Using the 
``RAJA::expt::launch method`` developers are provided with an execution 
space enabling them to express algorithms in terms of nested
``RAJA::expt::loop`` statements::

  RAJA::expt::launch<launch_policy>(select_CPU_or_GPU)
  RAJA::expt::Grid(RAJA::expt::Teams(NE), RAJA::expt::Threads(Q1D)),
  [=] RAJA_HOST_DEVICE (RAJA::expt::Launch ctx) {

    RAJA::expt::loop<team_x> (ctx, RAJA::RangeSegment(0, teamRange), [&] (int bx) {

      RAJA_TEAM_SHARED double s_A[SHARE_MEM_SIZE];

      RAJA::expt::loop<thread_x> (ctx, RAJA::RangeSegment(0, threadRange), [&] (int tx) {
        s_A[tx] = tx;
      });

        ctx.teamSync();

   )};

  });
  
The underlying idea of *RAJA Launch* is to enable developers to express hierarchical
parallelism in terms of teams and threads. Similar to the CUDA programming model,
development is done using a collection of threads, threads are grouped into teams.
Using the ``RAJA::expt::loop`` methods iterations of the loop may be executed by threads
or teams (depending on the execution policy). The launch context serves to synchronize
threads within the same team. The *RAJA Launch* abstraction consist of three main concepts.

  * *Launch Method*: creates an execution space in which developers may express 
    their algorithm in terms of nested ``RAJA::expt::loop`` statements. The loops are then
    executed by threads or thread-teams. The method is templated on both a host
    and device execution space and enables run-time selection of the execution environment.

  * *Resources*: holds a number of teams and threads (akin to CUDA blocks/threads).

  * *Loops*: are used to express hierarchical parallelism. Work within a loop is mapped to either teams or threads. Team shared memory
    is available by using the ``RAJA_TEAM_SHARED`` macro. Team shared memory enables
    threads in a given team to share data. In practice, team policies are typically
    aliases for RAJA GPU block policies in the x,y,z dimensions (for example cuda_block_direct),
    while thread policies are aliases for RAJA GPU thread policies (for example cuda_thread_direct)
    x,y,z dimensions. On the host, teams and threads may be mapped to sequential
    loop execution or OpenMP threaded regions.

The team loop interface combines concepts from ``RAJA::forall`` and ``RAJA::kernel``.
Various policies from ``RAJA::kernel`` are compatible with the ``RAJA::launch``
framework.

.. _loop_elements-CombiningAdapter-label:

--------------------------------
MultiDimensional loops using Simple loop APIs (RAJA::CombiningAdapter)
--------------------------------

A ``RAJA::CombiningAdapter`` object provides ways to run perfectly nested loops
with simple loop APIs like ``RAJA::forall`` and ``RAJA::WorkGroup`` :ref:`workgroup-label`.
To introduce the ``RAJA ::CombiningAdapter`` interface, consider a (N+1)-level
C-style loop nest::

  for (int iN = 0; iN < NN; ++iN) {
    ...
       for (int i0 = 0; i0 < N0; ++i0) {
         \\ inner loop body
       }
  }

We can use a ``RAJA::CombiningAdapter`` to combine the iteration spaces of the
loops and pass the adapter to a ``RAJA::forall`` statement to execute them::

  auto adapter = RAJA::make_CombingingAdapter(
      [=] (int iN, ..., int i0)) {
        \\ inner loop body
      }, IN, ..., I0);

  RAJA::forall<exec_policy>(adapter.getRange(), adapter);

A ``RAJA::CombiningAdapter`` object is a template combining a loop body and
iteration spaces. The maker function template takes a lambda expression for the
loop body and an arbitrary number of segment arguments. It provides a flattened
index space via the ``getRange`` method that can be passed as the iteration space
to the simple loop API. The object itself can be passed into the loop API as the
loop body. The object's call operator does the conversion of the flat single
dimensional index into the multi-dimensional index space, calling the provided
lambda with the appropriate indices.

.. note:: CombiningAdapter currently only supports ``RAJA::RangeSegment`` and
          ``RAJA::TypedRangeSegment`` segments.
