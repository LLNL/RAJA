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

.. _loop_elements-label:

==============================================
Elements of Loop Execution
==============================================

In this section, we describe the basic elements of RAJA loop kernel execution. 
``RAJA::forall`` and ``RAJA::kernel`` template methods comprise the
RAJA interface for loop execution. ``RAJA::forall`` methods execute simple 
loops (e.g., non-nested loops) while ``RAJA::kernel`` methods support nested 
loops and other complex loop kernels and transformations.

.. note:: * All **forall** and **kernel** methods are in the namespace ``RAJA``.
          * A ``RAJA::forall`` loop execution method is a template on an 
            *execution policy* type. A ``RAJA::forall`` method takes two 
            arguments: 
              * an iteration space object, and
              * a lambda expression representing the loop body.
          * Each ``RAJA::kernel`` method is a template on a policy that 
            contains statements with *execution policy* types appropriate for 
            the kernel structure; e.g., an execution policy for each level in a
            loop nest. A ``RAJA::kernel`` method takes multiple arguments:
              * a *tuple* of iteration space objects, and
              * one or more lambda expressions representing portions of 
                the loop kernel body.

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

  double* a = ...;
  double* b = ...;
  double* c = ...;
  
  for (int i = 0; i < N; ++i) {
    c[i] = a[i] + b[i];
  }

may be written in a RAJA form as::

  double* a = ...;
  double* b = ...;
  double* c = ...;
  
  RAJA::forall<exec_policy>(iter_space I, [=] (index_type i) {
    c[i] = a[i] + b[i];
  });

A ``RAJA::forall`` method is a template on an execution policy type and takes
two arguments: an object describing the loop iteration space, such as a RAJA 
segment or index set, and a lambda expression for the loop body. Applying 
different loop execution policies enables the loop to run in different ways; 
e.g., using different programming model back-ends. Different iteration space 
objects enable the loop iterates to be partitioned, reordered, run in 
different threads, etc. 

.. note:: Changing loop execution policy types and iteration space constructs
          enable loops to run in different ways by recompiling the code and 
          without modifying the loop kernel code.

While loop execution using ``RAJA::forall`` methods is a subset of 
``RAJA::kernel`` functionality, described next, we maintain the 
``RAJA::forall`` interface for simple loop execution because the syntax is 
simpler and less verbose.

.. note:: Data arrays in lambda expressions used with RAJA are typically 
          RAJA Views (see :ref:`view-label`) or bare pointers as shown in
          the code snippets above. Using something like 'std::vector' is
          non-portable (won't work in CUDA kernels) and would add excessive 
          overhead for copying data into the lambda data environment.

.. _loop_elements-kernel-label:

----------------------------
Complex Loops (RAJA::kernel)
----------------------------

A ``RAJA::kernel`` template provides ways to compose and execute arbitrary 
loop nests and other complex kernels. To introduce the RAJA *kernel* interface,
consider a (N+1)-level C-style loop nest::

  for (index_type iN = 0; iN < NN; ++iN) {
    ...
       for (index_type i0 = 0; i0 < N0; ++i0) {s
         \\ inner loop body
       }
  }

Note that we could write this by nesting ``RAJA::forall`` statements and
it would work, assuming the execution policies were chosen properly::

  RAJA::forall<exec_policyN>(IN, [=] (index_type iN) {
    ...
       RAJA::forall<exec_policy0>(I0, [=] (index_type i0)) {
         \\ inner loop body
       }
    ...
  }

However, this approach treats each loop level as an independent entity. This
makes it difficult to parallelize the levels in the loop nest together. So it
limits the amount of parallelism that can be exposed and the types of 
parallelism that may be used. For example, if an OpenMP or CUDA
parallel execution policy is used on the outermost loop, then all inner loops
would be run sequentially in each thread. It also makes it difficult to perform 
transformations like loop interchange and loop collapse. 

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
  
    RAJA::kernel< KERNEL_POL >(
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
``RAJA::statement::Lambda<0>`` type indicating that the first lambda expression
(argument zero of the sequence of lambdas passed to the ``RAJA::kernel`` method)
will comprise the inner loop body. We only have one lambda in this example 
but, in general, we can have any number of lambdas and we can use any subset 
of them, with ``RAJA::statement::Lambda`` types placed appropriately in the 
execution policy, to construct a loop kernel. For example, placing 
``RAJA::statement::Lambda`` types between ``RAJA::statement::For`` statements 
enables non-perfectly nested loops.

Each lambda expression passed to a ``RAJA::kernel`` method **must take an 
index argument for each iteration space in the tuple**. However, any subset 
of the arguments may actually be used in each lambda expression. 

.. note:: The loop index arguments for each lambda expression used in a RAJA 
          kernel loop body **must match** the contents of the 
          *iteration space tuple* in number, order, and type. Not all index 
          arguments must be used in each lambda, but they **all must appear** 
          for the RAJA kernel to be well-formed. In particular, your code will 
          not compile if this is not done correctly. If an argument is unused
          in a lambda expression, you may include its type and omit its name
          in the argument list to avoid compiler warnings just as one would do
          for a regular C++ method.

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

A summary of all RAJA execution policies that may be used with ``RAJA::forall``
or ``RAJA::kernel`` may be found in :ref:`policies-label`. Also, a discussion
of how to construct ``RAJA::KernelPolicy`` types and available 
``RAJA::statement`` types can be found in :ref:`loop_elements-kernelpol-label`.
