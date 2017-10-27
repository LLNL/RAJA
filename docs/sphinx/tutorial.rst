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
about C++ lambda expressions, which are typically used with RAJA constructs.

===============================
A Little C++ Lambda Background
===============================

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
that are used in the lambda are captured by value or reference, respectively.

=========
Examples
=========

The remainder of this tutorial illustrates how to use RAJA with examples that
use common numerical algorithm patterns. Note that all the examples employ
RAJA traversal template methods. For more information about them, please see
:ref:`forall-label`.

.. toctree::
   :maxdepth: 1

   tutorial/addVectors.rst
   tutorial/matrixMultiply.rst
   tutorial/jacobi.rst
   tutorial/wave.rst
   tutorial/gaussSeidel.rst
