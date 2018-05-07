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

The ``RAJA::forall`` and ``RAJA::kernel`` loop traversal template 
methods are the building blocks for most RAJA usage involving loop execution. 
``RAJA::forall`` methods are used to execute simple loops (typically
non-nested loops). ``RAJA::kernel`` methods support nested loops and 
complex transformations of loop kernels.

RAJA users pass application code fragments, representing loop bodies, into 
these traversal methods as lambda expressions. Iteration space objects, which
describe loop indexing, are also passed. Once in proper RAJA form, a loop
kernel can be run in different ways (e.g., using different programming model 
back-ends or changing how a loop iteration space is traversed) by changing 
execution policy template arguments or iteration space objects. Such changes
can be made without altering the loop kernel code itself.

.. note:: * All **forall** and **kernel** methods are in the namespace ``RAJA``.
          * Each ``RAJA::forall`` loop traversal method is templated on an 
            *execution policy* type. A 'forall' method takes two arguments: an 
            iteration space object and lambda expression kernel body.
          * Each ``RAJA::kernel`` method is templated on a policy that contains 
            statements with *execution policy* types appropriate for the
            kernel structure; e.g., an execution policy for each level in a
            loop nest. A 'kernel' method takes a *tuple* of iteration space
            objects and one or more lambda expressions representing parts of
            loop bodies.

Various examples showing how to use ``RAJA::forall`` and ``RAJA::kernel`` 
methods may be found in the :ref:`tutorial-label`.

For more information on RAJA execution policies and iteration space constructs, 
see :ref:`policies-label` and :ref:`index-label`, respectively. 

.. _loop_elements-forall-label:

-------------------------
Simple (Non-nested) Loops
-------------------------

As noted earlier, the ``RAJA::forall`` template is used to execute simple 
(e.g., non-nested) loops. For example, a C-style loop like::

  for (int i = 0; i < N; ++i) {
    c[i] = a[i] + b[i];
  }

may be written in a RAJA form as::

  RAJA::forall<exec_policy>(iter_space I, [=] (index_type i)) {
    c[i] = a[i] + b[i];
  });

``RAJA::forall`` templates take a template argument for the 
execution policy and two arguments: an object describing the loop iteration 
space, such as a RAJA segment or index set, and a lambda expression defining 
the loop body. Applying different loop execution policies enables the loop to 
run in different ways; e.g., using different programming model back-ends. 
Different iteration space objects, enable the loop iterates to be reordered, 
run in different threads, etc. 

.. note:: Changing loop execution policy types and iteration space constructs
          enable loops to run in different ways without modifying the loop 
          kernel code.

.. _loop_elements-nested-label:

-------------------------
Nested Loops
-------------------------

``RAJA::kernel`` templates provide ways to transform and execute arbitrary 
loop nests as we briefly describe here. A (N+1)-level loop nest, such as::

  for (int iN = 0; iN < NN; ++iN) {
    ...
       for (int i0 = 0; i0 < N0; ++i0) {s
         \\body
       }
  }

may be written in a RAJA form as::

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
         //loop body
      }

    );

Here, the ``RAJA::kernel`` template takes a ``RAJA::KernelPolicy`` template 
argument that defines a nested sequence of ``RAJA::statement::For`` types, 
one for each level in the loop nest. Each 'For' statement has an integral 
template argument (described below) and an execution policy template argument 
for the corresponding loop nest level. The innermost type in the kernel 
policy is a ``RAJA::statement::Lambda`` type indicating which lambda 
(or lambdas, in general) will comprise the inner loop body.

.. note:: The nesting of ``RAJA::statement::For`` types is analogous to how one
          would nest for-statements in a traditional C-style loop nest.

The first argument to the ``RAJA::kernel`` method is a tuple of N+1 iteration 
spaces, one for each loop nest level. This argument is followed by one or more 
lambda expression arguments that are used to form the inner loop body. Here, 
we have only one lambda expression argument that will be executed as the inner 
loop body; this is indicated by the ``RAJA::statement::Lambda<0>`` innermost 
type in the kernel policy above (i.e., '0' refers to the first lambda in
the argument list).

.. note:: The arguments for each lambda expression that is used in a RAJA 
          kernel loop body are indices that **must match** the contents of the 
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

For discussion of advanced loop construction and transformations using 
``RAJA::kernel``, along with examples, please see :ref:`complex_intro-label`

In summary, these RAJA template methods for loop kernel execution described
here require a user to have a basic understanding of specify several RAJA
concepts:

  #. Execution policies.

  #. The loop iteration space(s) -- often, an iteration space can be any valid random access container allowing users to define their own iteration space types.

  #. The lambda capture type; e.g., [=] or [&].

  #. Writing loop bodies as lambda expressions.

  #. Loop iteration variables and types, which are arguments to a lambda loop body.
