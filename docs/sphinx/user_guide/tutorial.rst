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

This RAJA tutorial introduces most commonly-used RAJA concepts and
capabilities via a sequence of examples. To understand, the discussion and
code examples, a working knowledge of C++ templates and lambda functions
is required. Before we begin, we provide a bit of background discussion of
the key features of C++ lambda expressions, which are essential using RAJA
easily.

===============================
A Little C++ Lambda Background
===============================

RAJA is used most easily and effectively by employing C++ lambda expressions,
especially for the bodies of loop kernels. Lambda expressions were 
introduced in C++11 to provide a lexically-scoped name binding; i.e., a 
*closure* that stores a function with a data environment. In particular, a 
lambda has the ability to *capture* variables from an enclosing scope for use 
within the local scope of a a function. 

Here, we provide a brief description of the essential elements of C++ lambdas.
A more technical and detailed discussion is available here:
`Lambda Functions in C++11 - the Definitive Guide <https://www.cprogramming.com/c++11/c++11-lambda-closures.html>`_ 

A C++ lambda expression takes the following form::

  [capture list] (parameter list) {function body}

The ``capture list`` defines how variables outside the lambda scope are pulled
into the lambda data environment. The ``parameter list`` defines arguments 
passed to the lambda function body; i.e., just like defining arguments to
a standard C++ method. RAJA template methods pass arguments to lambdas 
based on usage and context. Values in the capture list are initialized when 
the lambda is created, while values in the parameter list are set when the 
lambda function is called. The body of a lambda functions similarly to the 
body of an ordinary C++ method.

A C++ lambda can capture values in the capture list by value or by reference.
This is similar to how arguments to C++ methods are passed; e.g., 
pass-by-reference, pass-by-value, etc. However, there are some subtle 
differences between lambda variable Variables mentioned in the capture 
list with no extra symbols are captured by value. Capture by reference is
accomplished by using the common reference symbol '&'; for example::

  int x;
  int y = 100;
  [&x, &y](){ x = y; };

generates a lambda that captures 'x' and 'y' by reference and assigns the 
value of 'y' to 'x' when called. The same outcome would be achieved by 
writing::

  [&](){ x = y; };   // capture all lambda arguments by reference...

or::

  [=, &x](){ x = y; };  // capture 'x' by reference and 'y' by value...  

Note that the following two attempts will generate compilation errors::

  [=](){ x = y; };      // capture all lambda arguments by value...
  [x, &y](){ x = y; };  // capture 'x' by value and 'y' by reference...

It is illegal to assign a value to a variable 'x' that is captured
by value; i.e.,  it is `read-only`.

.. note:: For most RAJA usage, it is recommended to capture all lambda 
          variables `by-value`. This is required for executing a RAJA loop 
          on a device, such as a GPU, and doing so will allow the code to 
          be portable for either CPU or GPU execution.


=========
Examples
=========

The remainder of this tutorial illustrates how to exercise various RAJA 
features using simple examples. Note that all the examples employ
RAJA traversal template methods, which are described briefly here :ref:`forall-label`.

.. toctree::
   :maxdepth: 1

   tutorial/addVectors.rst
   tutorial/matrixMultiply.rst
   tutorial/jacobi.rst
   tutorial/wave.rst
   tutorial/gaussSeidel.rst
