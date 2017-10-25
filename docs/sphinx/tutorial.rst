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

.. _tutorial-label:

**********************
Tutorial
**********************

This section contains a RAJA tutorial that introduces RAJA concepts and
capabilities via examples. First, we provide some background discussion 
about C++ lambda expressions and RAJA traversal template methods, which
are used out the examples.

====================
C++ Lambda Overview
====================

RAJA is used most easily and effectively by employing C++ lambda expressions. 
Here, we provide a brief description of the essential elements of C++ lambdas.
`Lambda expressions <http://en.cppreference.com/w/cpp/language/lambda>`_ were 
introduced in C++11 to provide a lexically-scoped name binding; i.e., a 
*closure* that stores a function with a data environment. In particular, a 
lambda has the ability to *capture* variables from an enclosing scope for use 
within the local scope of a a function. 

A lambda expression takes the following form::

  [capture list] (parameter list) {function body}

The capture list corresponds to variables outside the lambda, while the 
parameter list defines arguments to the lambda function body. Values in the 
capture list are initialized when the lambda is created, while values in the 
parameter list are initialized when the lambda function is called. A lambda 
can capture values in the capture list by value or by reference, which is 
similar to standard C++ function arguments. Variables mentioned in the capture 
list with no extra symbols are captured by value. Capture by reference may be 
accomplished using the '&' symbol; for example::

  int x;
  int y = 100;
  [&x, &y](){x=y;]

generates a lambda that assigns the value of 'y' to 'x' when called. By 
setting the capture list as ``[=]`` or ``[&]`` all variables in scope 
that are used in the lambda are captured by value or reference, respectively.

=========================
RAJA Traversal Templates
=========================

RAJA users typically pass application code fragments, such as loop bodies,
into RAJA loop traversal template methods using lambda expressions. The two 
main types of RAJA traversal templates are ``RAJA::forall`` and
``RAJA::forallN``. Once loops are written using these constructs, they can
be run using different programming model back-ends by changing template
parameters.

The ``RAJA::forall`` templates abstract standard C-style for loops. They are
templated execution policies and take loop iteration spaces and lambdas 
defining loop bodies as arguments; e.g.,::

  RAJA::forall<exec_policy>(iter_space I, [=] (index_type i)) {
    // loop body
  });

The ``RAJA::forallN`` traversal templates provide flexibility in
how arbitrary loop nests can be run with minimal source code changes. A
``RAJA::ForallN`` loop is templated on 'N' execution policy and iteration
space parameters, one for each level in the loop nest, plus a lambda for the
inner loop body; e.g.,::

  RAJA::forallN< RAJA::NestedPolicy<
                 RAJA::ExecList< exec_policy1, .... , exec_policyN> > >(
    iter_space I1,..., iter_space IN, 
    [=](index_type i1,..., index_type iN) {
      // loop body
  });

In summary, these RAJA template methods require a user to understand how to
specify several items:

  #. The lambda capture type; e.g., [=] or [&]

  #. The desired execution policy (or policies)

  #. The loop iteration space(s) -- in most cases an iteration space can be any valid random access container

  #. The data type of loop iteration variables

  #. The lambda that defines the loop body

The remainder of this tutorial illustrates how to use RAJA with examples that
use common numerical algorithm patterns.

=========
Examples
=========

.. toctree::
   :maxdepth: 1

   tutorial/addVectors.rst
   tutorial/matrixMultiply.rst
   tutorial/jacobi.rst
   tutorial/wave.rst
   tutorial/gaussSeidel.rst
