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

=====================
Overview
=====================

As RAJA is built on the C++ lambda, we provide a brief overview on lambda functions.
Lambda's were introduced to allow for the construction of in place functions. A lambda has the ability to "capture" variables from a local context for use within the function. A lambda expression takes the following form

.. code-block:: cpp

  [capture list] (parameter list) {function body}

The capture list corresponds to variables outside the lambda, while the parameter list defines the function arguments to the lambda function. The values in the capture list are initialized when the lambda is created, while values in the parameter list are initialized when the lambda function is called. A lambda can capture values in the capture list by copy or by reference. Variables mentioned in the capture list with no extra symbols are captured by value. Capture by reference may be accomplished using the & symbol; for example

.. code-block:: cpp

  int x;
  int y = 100;
  [&x, &y](){x=y;]

will generate a lambda which will assign the value of y to x when called. By setting the capture list as ``[=]`` or ``[&]`` all variables within scope which are used in the lambda will be captured by copy or reference respectively.

Building from the C++ lambda, RAJA introduces two types of templated methods, namely ``RAJA::forall`` and ``RAJA::forallN``. The ``RAJA::forall`` method is an abstraction of the standard C for loop. It is templated on an execution policy and takes an iteration space and a lambda capturing the loop body as arguments.

.. code-block:: cpp

  RAJA::forall<exec_policy>(iter_space I, [=] (index_type i)) {
    //body
  });

Similarly, the ``RAJA::ForallN`` loop is an abstraction of nested ``for`` loops. The ``RAJA::ForallN`` loop is templated on up to N execution policies and expects an iteration space for each execution policy and a lambda with an argument for each iteration space.

.. code-block:: cpp

  RAJA::forallN<
    RAJA::NestedPolicy<exec_policy1, .... , exec_policyN> >(
      iter_space I1,..., iter_space IN, [=](index_type i1,..., index_type iN) {
         //body
  });

In summary, using one of the RAJA templated methods requires the developer to supply the following
1. Capture type - [=] or [&]
3. exec_policy  - How the traversal occurs
4. iter_space   - An iteration space for the RAJA loop (any random access container)
5. index_type   - Type of values contained in the iteration space
6. lambda       - The body of the loop

The remainder of the tutorial demonstrates the utility of RAJA by drawing from commonly used computing patterns.



--------
Examples
--------

.. toctree::
   :maxdepth: 1

   tutorial/addVectors.rst
   tutorial/matrixMultiply.rst
   tutorial/jacobi.rst
   tutorial/wave.rst
   tutorial/gaussSeidel.rst
