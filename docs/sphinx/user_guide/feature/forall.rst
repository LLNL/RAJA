.. ##
.. ## Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
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

.. _forall-label:

=========================
forall and nested::forall
=========================

The ``forall`` and ``nested::forall`` loop traversal template methods are 
the building block for most RAJA usage. RAJA users pass application 
code fragments, such as loop bodies, into these loop traversal methods 
(using lambda expressions, for example) along with iteration space
information. Then, once loops are written in the RAJA form, they can
be run using different programming model back-ends by changing template
arguments for the execution policies. For information on available RAJA
execution policies, see :ref:`policies-label`.

.. note:: * All RAJA forall and nested::forall methods are in the namespace ``RAJA``.
          * Each loop traversal method is templated on an *execution policy*,
            or multiple execution policies for the case of ``nested::forall``.

The ``RAJA::forall`` templates abstract standard C-style for loops.  
For example, a C-style loop like::

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

The ``RAJA::nested::forall`` traversal templates provide flexibility in
how arbitrary loop nests can be run with minimal source code changes. A
loop nest, such as::

  for (int iN = 0; iN < NN; ++iN) {
    ...
       for (int i0 = 0; i0 < N0; ++i0) {s
         \\body
       }
  }

may be written in a RAJA form as::
  
    RAJA::nested::forall< RAJA::nested::Policy<
                      RAJA::nested::For<N, exec_policyN>, ...
                      RAJA::nested::For<0, exec_policy0>,
		      RAJA::make_tuple(iter_space IN, ..., iter_space I0),
                      [=] (index_type iN, ... , index_type i1) {
                      //loop body
    });

A ``RAJA::nested::forall`` method is templated on 'N+1' execution policy arguments,
and takes N+1 iteration space arguments, one for each level in the loop nest, 
plus a lambda expression for the inner loop body.

.. note:: For the nested loop case, the parameters for the execution policies, 
          loop-level iteration spaces, and the arguments to the lambda must 
          all be in the same order, starting with the outermost loop and ending
          with the innermost loop when read from left to right.

In summary, these RAJA template methods require a user to understand how to
specify several items:

  #. The desired execution policy (or policies).

  #. The loop iteration space(s) -- in most cases an iteration space can be any valid random access container.

  #. The lambda capture type; e.g., [=] or [&].

  #. The lambda expression that defines the loop body.

  #. The loop iteration variables and their types, which are arguments to the lambda loop body.

Common usage of ``RAJA::forall`` may be found in the examples in 
:ref:`tutorial-label`.
