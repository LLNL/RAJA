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

.. _loop_elements-label:

==============================================
Elements of Loop Execution
==============================================

``RAJA::forall`` and ``RAJA::kernel`` template methods comprise the
RAJA interface for loop execution. ``RAJA::forall`` methods execute simple 
loops (e.g., non-nested loops). ``RAJA::kernel`` methods support nested loops 
and other complex loop kernels.

.. note:: * All **forall** and **kernel** methods are in the namespace ``RAJA``.
          * A ``RAJA::forall`` loop traversal method is templated on an 
            *execution policy* type. A 'forall' method takes two arguments: 
              * an iteration space object, and
              * a lambda expression representing the loop body.
          * Each ``RAJA::kernel`` method is templated on a policy that contains 
            statements with *execution policy* types appropriate for the
            kernel structure; e.g., an execution policy for each level in a
            loop nest. A 'kernel' method takes multiple arguments:
              * a *tuple* of iteration space objects, and
              * one or more lambda expressions representing portions of 
                the loop kernel body.

In this section we describe the basic elements of how these methods work. 
Various examples showing how to use ``RAJA::forall`` and ``RAJA::kernel`` 
methods may be found in the :ref:`tutorial-label`.

For more information on RAJA execution policies and iteration space constructs, 
see :ref:`policies-label` and :ref:`index-label`, respectively. 

.. _loop_elements-forall-label:

---------------------------
Simple Loops (RAJA::forall)
---------------------------

As noted earlier, a ``RAJA::forall`` template executes simple 
(e.g., non-nested) loops. For example, a C-style loop like::

  for (int i = 0; i < N; ++i) {
    c[i] = a[i] + b[i];
  }

may be written in a RAJA form as::

  RAJA::forall<exec_policy>(iter_space I, [=] (index_type i)) {
    c[i] = a[i] + b[i];
  });

A ``RAJA::forall`` method is templated on an execution policy type and takes
two arguments: an object describing the loop iteration space, such as a RAJA 
segment or index set, and a lambda expression for the loop body. Applying 
different loop execution policies enables the loop to run in different ways; 
e.g., using different programming model back-ends. Different iteration space 
objects enable the loop iterates to be partitioned, reordered, run in 
different threads, etc. 

.. note:: Changing loop execution policy types and iteration space constructs
          enable loops to run in different ways without modifying the loop 
          kernel code.

While loop execution using ``RAJA::forall`` methods is a subset of 
``RAJA::kernel`` functionality, described next, we maintain the 
``RAJA::forall`` interface for simple loop execution because the syntax is 
simpler and less verbose.

.. _loop_elements-kernel-label:

----------------------------
Complex Loops (RAJA::kernel)
----------------------------

A ``RAJA::kernel`` template provides ways to transform and execute arbitrary 
loop nests and other complex kernels. To introduce the RAJA *kernel* interface,
consider a (N+1)-level C-style loop nest::

  for (index_type iN = 0; iN < NN; ++iN) {
    ...
       for (index_type i0 = 0; i0 < N0; ++i0) {s
         \\ inner loop body
       }
  }

This could be written by nesting ``RAJA::forall`` statements::

  RAJA::forall<exec_policyN>(IN, [=] (index_type iN)) {
    ...
       RAJA::forall<exec_policy0>(I0, [=] (index_type i0)) {
         \\ inner loop body
       }
    ...
  }

However, this approach treats each loop level as an independent entity, which
makes it difficult to perform transformations like loop interchange and
loop collapse. It also limits the amount of parallelism that can be exposed or
the type of parallelism that may be used. For example, if an OpenMP or CUDA
parallel execution policy is used on the outermost loop, then all inner loops
would be run in each thread.

The RAJA *kernel* interface facilitates parallel execution and transformations 
of arbitrary loop nests and other complex loops. It can treat a complex loop 
structure as a single entity, which simplifies the ability to apply kernel
transformations and different parallel execution patterns by changing one 
execution policy type.

The loop nest may be written in a RAJA kernel form as::

    using KERNEL_POL = 
      RAJA::KernelPolicy< RAJA::statement::For<N, exec_policyN, 
                            ...
                              RAJA::statement::For<0, exec_policy0,
                                RAJA::statement::Lambda<0>
                              >
                            ...
                          > 
                        >;
  
    RAJA::kernel< KERNEL_POLICY >(
      RAJA::make_tuple(iter_space IN, ..., iter_space I0),

      [=] (index_type iN, ... , index_type i0) {
         // inner loop body
      }

    );

A ``RAJA::kernel`` method takes a ``RAJA::KernelPolicy`` type template 
parameter, and a tuple of iteration spaces and a sequence of lambda 
expressions as arguments. 

In the case we discuss here, the execution policy contains a nested sequence
of ``RAJA::statement::For`` statements, one for each level in the loop nest. 
Each 'For' statement takes three template parameters: 

  * an integral index parameter that binds it to the item in the iteration 
    space tuple associated with that index,
  * an execution policy type for the corresponding loop nest level, and
  * an *enclosed statement list* (described in :ref:`loop_elements-kernelpol-label`).

.. note:: The nesting of ``RAJA::statement::For`` types is analogous to the
          nesting of for-statements in the C-style version of the loop nest.
          A notable syntactic difference is that curly braces are replaced 
          with '<, >' symbols enclosing the template parameter lists.

Here, the innermost type in the kernel policy is a 
``RAJA::statement::Lambda<0>`` type indicating that the first lambda 
(lambda argument zero) will comprise the inner loop body. We only have one
lambda in this example but, in general, we can have any number of lambdas
and we can use any subset of them, with ``RAJA::statement::Lambda`` types
placed appropriately in the execution policy, to construct a loop kernel.
For example, placing ``RAJA::statement::Lambda`` types between 
``RAJA::statement::For`` statements enables non-perfectly nested loops.

Each lambda passed to a RAJA kernel method **must take an index argument for 
each iteration space in the tuple**. However, any subset of the arguments may 
actually be used in each lambda expression. 

.. note:: The loop index arguments for each lambda expression used in a RAJA 
          kernel loop body **must match** the contents of the 
          *iteration space tuple* in number, order, and type. Not all index 
          arguments must be used in each lambda, but they all must appear for
          the RAJA kernel to be well-formed.

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
loop functionality and :ref:`nestedreorder-label` for a detailed example 
describing nested loop reordering.

.. _loop_elements-kernelpol-label:

--------------------------------
RAJA Kernel Execution Policies
--------------------------------

RAJA kernel policies are constructed with a combination of *Statements* and
*Statement Lists* that forms a simple domain specific language that
relies **solely on standard C++11 template support**. A Statement is an 
action, such as executing a loop, invoking a lambda, setting a thread barrier, 
etc. A StatementList is an ordered list of Statements that are executed 
sequentially. A Statement may contain an enclosed StatmentList. Thus, a 
``RAJA::KernelPolicy`` type is simply a StatementList.

The main Statements types provided by RAJA are ``RAJA::statement::For`` and
``RAJA::statement::Lambda``, that we discussed above. A 'For' Statement 
indicates a for-loop structure and takes three template arguments: 
'ArgId', 'ExecPolicy', and 'EnclosedStatements'. The ArgID identifies the 
position of the corresponding iteration space in the tuple argument to the 
``RAJA::kernel`` method. The ExecPolicy gives the RAJA execution policy to 
use on the loop (similar to using ``RAJA::forall``). EnclosedStatements 
contain whatever is nested within the template parameter list and form a 
StatementList, which is executed for each iteration of the loop.
The ``RAJA::statement::Lambda<LambdaID>`` invokes the lambda corresponding to
its position (LambdaID) in the ``RAJA::kernel`` argument list. For example,
a simple sequential for-loop::

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

  * ``RAJA::statement::For< ArgId, ExecPolicy, EnclosedStatements >`` abstracts a for-loop associated with kernel iteration space with tuple index 'ArgId', to be run with 'ExecPolicy' execution policy, and containing the 'EnclosedStatements' which are executed for each loop iteration.
  * ``RAJA::statement::Lambda< LambdaId >`` invokes the lambda expression that appears at index 'LambdaId' in the sequence of lambda arguments to a kernel.
  * ``RAJA::statement::Collapse< ExecPolicy, ArgList<...>, EnclosedStatements >`` collapses multiple perfectly nested loops specified by tuple iteration space indices in 'ArgList', using the 'ExecPolicy' execution policy, and places 'EnclosedStatements' inside the collapsed loops which are executed for each iteration.
  * ``RAJA::statement::If< Conditional >`` chooses which portions of a policy to run based on run-time evaluation of conditional statement; e.g., true or false, equal to some value, etc. 
  * ``RAJA::statement::CudaKernel< EnclosedStatements>`` launches 'EnclosedStatements' as a CUDA kernel; e.g., a loop nest where iteration space of each loop level are associated to threads and/or thread blocks. 
  * ``RAJA::statement::CudaSyncThreads`` provides CUDA '__syncthreads' barrier; a similar thread barrier for OpenMP will be added soon.
  * ``RAJA::statement::Hyperplane< ArgId, HpExecPolicy, ArgList<...>, ExecPolicy, EnclosedStatements >`` provides a hyperplane iteration pattern over multiple indices.
  * ``RAJA::statement::SetShmemWindow< EnclosedStatements >`` sets a window into a shared memory buffer for loops described by 'EnclosedStatements'.
  * ``RAJA::statement::Tile< ArgId, TilePolicy, ExecPolicy, EnclosedStatements >`` creates tiling (or cache blocking) of outer loop associated with kernel iteration space with tuple index 'ArgId' for inner loops described by 'EnclosedStatements' using given 'TilePolicy' (e.g., fixed tile size) and 'ExecPolicy' execution policy.

Various examples that illustrate the use of these statement types can be found
in :ref:`complex_loops-label`.


