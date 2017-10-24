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

.. _tutorial-label::

**********************
Tutorial
**********************

This section provides a RAJA tutorial, introducing RAJA concepts and
capabilities via examples.

====================
C++ Lambda Overview
====================

RAJA is used most easily and effectively by employing C++ lamda expressions. 
Here, we provide a brief description of the eseential elements of C++ lambdas.
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

RAJA users typically pass application code frangments, such as loop bodies,
into RAJA loop traversal template methods using lambdas. The two main types
of RAJA traversal templates are ``RAJA::forall`` and ``RAJA::forallN``. The 
``RAJA::forall`` method abstracts a standard C-style for loop. It is templated 
on an execution policy and takes a loop iteration space and a lambda defining 
the loop body as arguments; e.g.,::

  RAJA::forall<exec_policy>(iter_space I, [=] (index_type i)) {
    //body
  });

Similarly, the ``RAJA::ForallN`` loop abstracts nested ``for`` loops. A
``RAJA::ForallN`` loop is templated on 'N' execution policy and iteration
space paramters, one for each level in the loop nest, plus a lambda for the
inner loop body; e.g.,::

  RAJA::forallN< RAJA::NestedPolicy<
                 RAJA::ExecList< exec_policy1, .... , exec_policyN> > >(
    iter_space I1,..., iter_space IN, 
    [=](index_type i1,..., index_type iN) {
      //body
  });

In summary, these RAJA template methods require a user to understand how to
specify the items::

  #. The lambda capture type; e.g., [=] or [&]
  #. The desired execution policy (or policies)
  #. The loop iteration space(s) - in most cases a valid iteration space is any
valid random access container.
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
