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

.. _loop_basic-label:

==============================================
Basic Loop Execution: Single and Nested Loops
==============================================

The ``RAJA::forall`` and ``RAJA::kernel`` loop traversal template 
methods are the building blocks for most RAJA usage involving loop execution. 
RAJA users pass application code fragments, representing loop bodies, into 
these traversal methods as lambda expressions. Iteration space objects, which
describe loop indexing, are also passed. Then, after loops are written in the 
RAJA form, they can be run using different programming model back-ends by 
changing execution policy template arguments. Index space traversal and
loop transformations can also be applied by changing iteration space objects
and execution policy details. For information on RAJA execution policies and
iteration space constructs, see :ref:`policies-label` and :ref:`index-label`,
respectively. For information about complex kernels and loop transformations,
please see :ref:`complex_intro-label`.

.. note:: * All **forall** and **kernel** methods are in the namespace ``RAJA``.
          * Each ``RAJA::forall`` traversal method is templated on an 
            *execution policy*. 
          * Each ``RAJA::kernel`` method requires a statement with an 
            *execution policy* type for each level in a loop nest.

-------------------------
Simple (Non-nested) Loops
-------------------------

The ``RAJA::forall`` templates is used to execute simple (e.g., non-nested) 
loops. For example, a C-style loop like::

  for (int i = 0; i < N; ++i) {
    c[i] = a[i] + b[i];
  }

may be written in a RAJA form as::

  RAJA::forall<exec_policy>(iter_space I, [=] (index_type i)) {
    c[i] = a[i] + b[i];
  });

The RAJA form takes a template argument for the execution policy, and
two arguments: an object describing the loop iteration space (e.g., a RAJA 
segment or index set) and a lambda expression defining the loop body.

-------------------------
Nested Loop Basics
-------------------------

The ``RAJA::kernel`` traversal templates provide flexibility in
how arbitrary loop nests can be run with minimal source code changes. A
(N+1)-level loop nest, such as::

  for (int iN = 0; iN < NN; ++iN) {
    ...
       for (int i0 = 0; i0 < N0; ++i0) {s
         \\body
       }
  }

may be written in a RAJA form as::
  
    RAJA::kernel< RAJA::KernelPolicy<

                    RAJA::statement::For<N, exec_policyN, 
                      ...
                        RAJA::statement::For<0, exec_policy0,
                          RAJA::statement::Lambda<0>
                        >
                      ...
                    > 
                >( 
      RAJA::make_tuple(iter_space IN, ..., iter_space I0),

      [=] (index_type iN, ... , index_type i0) {
         //loop body
    });

The ``RAJA::kernel`` template takes a ``RAJA::KernelPolicy`` template argument that
defines a nested sequence of ``RAJA::statement::For`` types, one for each level in 
the loop nest. Each one has an integral template argument (described below) and an 
execution policy template argument for the corresponding level in the loop nest. 
The innermost types in the kernel policy are ``RAJA::statement::Lambda`` types 
indicating the lambdas (or lambda) to be used to comprise the inner loop body.

.. note:: The nesting of ``RAJA::statement::For`` types is analogous to how one
          would nest for-statements in a traditional C-style loop nest.

The first argument to the ``RAJA::kernel`` method is a tuple of N+1 iteration spaces, 
one for each loop nest level. This argument is followed by one or more lambda 
expression arguments that may be used to form the inner loop body. Here, we have
only one lambda expression argument that will be executed as the inner loop body;
this is indicated by the ``RAJA::statement::Lambda<0>`` innermost type in the kernel
policy above.

.. note:: The arguments for each lambda expression that is used in a RAJA kernel 
          loop body are indices that must match the contents of the 
          *iteration space tuple* in number, order, and type. Not all index 
          arguments must be used in each lambda, but they all must appear.

For RAJA nested loops defined by a ``RAJA::kernel``, as shown above, the loop nest 
ordering is determined by the order of the nested policies, starting with the 
outermost loop and ending with the innermost loop. 

.. note:: The integer value that appears as the first parameter in each 
          ``RAJA::statement::For`` templates indicates which iteration space tuple
          entry or lambda index argument it corresponds to. **This allows loop 
          nesting order to be changed simply by changing the ordering of the 
          nested policy statements**. This is analogous to changing the order 
          of 'for-loop' statements in C-style nested loop code.

In summary, these RAJA template methods require a user to understand how to
specify several items:

  #. The desired execution policy (or policies).

  #. The loop iteration space(s) -- in most cases an iteration space can be any valid random access container.

  #. The lambda capture type; e.g., [=] or [&].

  #. The lambda expression that defines the loop body.

  #. The loop iteration variables and their types, which are arguments to the lambda loop body.

Basic usage of ``RAJA::forall`` and ``RAJA::kernel`` may be found 
in the examples in :ref:`tutorial-label`.
