.. ##
.. ## Copyright (c) 2016, Lawrence Livermore National Security, LLC.
.. ##
.. ## Produced at the Lawrence Livermore National Laboratory.
.. ##
.. ## All rights reserved.
.. ##
.. ## For release details and restrictions, please see raja/README-license.txt
.. ##


.. _tutorial::

========
Tutorial
========

This is an overview of the examples included in RAJA. In particular
we highlight RAJA features which simplify scientific computing.

At the heart of RAJA is the C++11 Lambda. Lambda functions were introduced to allow for the construction
in place functions. An underlying concept of lambda's is that variables maybe ``captured" from the local context
and used in the body of the loop. A lambda expression consist is of the following form

.. code-block:: cpp

   [capture list] (parameter list) {function body}


Here the capture and parameter list may be empty and thus the following is a valid lambda

.. code-block:: cpp

  [](){std::cout<<"RAJA Rocks!"<<std::endl;};


By default Lambda's capture by copy any variables within a block of code. Thus to modify values values the & symbol
must be added to the capture list. For example

.. code-block:: cpp

   int x;
   int y = 100;
   int istart = 0, iend = 10;
   [&x, &y](){x=y;]

will assign the value of y to x. Furthermore there are shortcuts for setting the capture type


1. [=] capture all variables within the block scope by copy
2. [&] capture all variables within the block scopy by reference


Building from the C++ lambda, RAJA introduces two main types of templated loops, namely
the ``RAJA::forall`` and ``RAJA::forallN`` loops. Here RAJA decouples a body loop
from its traversal. For example the ``RAJA::forall`` method is an abstracted version of the basic C++ loop.
The method is templated on an execution policies and takes an iteration space and a lambda which encapsulates
the loop body. 

.. code-block:: cpp
                
  RAJA::forall<exec_policy>(iter_space I, [=] (index_type i)) {
    //body
  });

Similarly the ``RAJA::ForallN`` loop is an abstraction of nested ``for`` loops. The ``RAJA::ForallN`` loop is
templated on up to N execution policies and takes in an iteration space and index for each execution policy.
  
.. code-block:: cpp
                
  RAJA::forallN<
    RAJA::NestedPolicy<exec_policy1, .... , exec_policyN> >(
      iter_space I1,..., iter_space IN, [=](index_type i1,..., index_type iN) {
         //body
  });
  
In each of these loops the developer must specify the following

1. [=] By-copy capture
2. [&] By-reference capture (for non-unified memory targets)
3. exec_policy - Specifies how the traversal occurs
4. iter_space  - Iteration space for RAJA loop (any random access container is expected)
5. index_type  - Index for RAJA loops

For the remainder of the tutorial we demonstrate the utility of RAJA by illustrating how
to create analogues of C++ style loops and highlighting features of RAJA which simplify
scientific computing.

---------------
Vector Addition
---------------
As a starting point we begin with simple vector addition. 
In this example, two vectors A, and B, of length N are added together.
The result is stored in a third array C. In standard C++ this may be carried out
as 

.. literalinclude:: ../../examples/example-add-vectors.cpp
                    :lines: 119-121
                            
Of course the standard C++ loop won't take advatage of the cores of multi/many-core processors,
but our RAJA analogue will! To construct the RAJA version we must first specify an execution policy
(more info see :ref:`ref-policy`) and construct an iteration space. For this example we can generate
an iteration space composed of a contiguous sequence of numbers by using the ``RAJA::RangeSegment``. 

.. literalinclude:: ../../examples/example-add-vectors.cpp
                    :lines: 132-137

By swapping out execution policies we can target different backends with the caveat being backends which offload
to devices. Here it is important that memory management is beyond the scope of RAJA (a seperate plug in is available
see :ref:`ref-plugins`) furthemore off loading to a device requires the ``__device__`` decorator on the lambda. 

.. literalinclude:: ../../examples/example-add-vectors.cpp
                    :lines: 163-168

As remark the CUDA exectution policy requires a number of threads per thread block.


A full working version ``example-add-vectors.cpp`` may be found in the example folder. 

---------------------
Matrix Multiplication
---------------------
As an example of nesting for loops we consider matrix multiplication.
Here we multiply two N x N matrices, A, and B. The result is then stored in C. 
Assuming that have pointers to the data

.. literalinclude:: ../../examples/example-matrix-multiply.cpp
                    :lines: 146-148

and with the aid of some macros

.. literalinclude:: ../../examples/example-matrix-multiply.cpp
                    :lines: 132-134

a C++ version of matrix multiplcation takes the form of 

.. literalinclude:: ../../examples/example-matrix-multiply.cpp
                    :lines: 161-171

With minimal effort we can start introducing RAJA into the algorithm.
First we can relive the need of macros by making use of ``RAJA::View``, which
simplifies multi-dimensional indexing (for more info see :ref:`ref-view`). 
                           
.. literalinclude:: ../../examples/example-matrix-multiply.cpp
                    :lines: 180-182

Second we can convert the outermost loop into a ``RAJA::forall`` loop

.. literalinclude:: ../../examples/example-matrix-multiply.cpp
                    :lines: 192-205

resulting in code that can be paired with different execution policies.
In the case the user will not offload to a device ``RAJA::forall`` loops
may be nested.

.. literalinclude:: ../../examples/example-matrix-multiply.cpp
                    :lines: 212-226  

As a generalization of nested loops, RAJA introduces the ``RAJA::forallN`` loop
which collapses a finite number of nested loops. Basic usage of ``RAJA::forallN``
requires an execution list ``RAJA::ExecList<>`` for the
``RAJA::NestedPolicy<>`` (for more info see :ref:`ref-nested`) . Each execution policy encapsulates how each loop should be
traversed. In the following example we pair the outerloop with an OpenMP policy and the inner loop with a sequential policy. 

.. literalinclude:: ../../examples/example-matrix-multiply.cpp
                    :lines: 254-264

A full working version ``example-matrix-multiply.cpp`` may be found in the example folder.
                            
-------------
Jacobi Method
-------------
Branching out to scientific computing we consider solving the following boundary value problem

.. math::
   
  U_{xx} + U_{yy} &= f, \quad U \in (0,1) \times (0,1), \\
  U(0,y) = U(1,y) &= 0, \\
  U(x,0) = U(x,1) &= 0,

where

.. math::

  f = 2x(y-1)(y-2x+xy+2) e^{(x-y)} .

To discretize the equation we consider the following
difference approximations on a structured grid

.. math::
   
   U_{xx} \approx \frac{U_{i+1,j} - 2U_{i,j} + U_{i-1,j}}{(\Delta x)^2}, \\
   U_{yy} \approx \frac{U_{i,j+1} - 2U_{i,j} + U_{i,j-1}}{(\Delta y)^2},

where (i,j) corresponds to a location on grid. 

   
-------------
Wave Equation
-------------
In this example we create a wave propagator which solves the
acoustic wave equation

.. math::  
   p_{tt} = c^{2} \left( p_{xx} + p_{yy} \right), \\
   (x,y) \in [0,1] \times [0,1].

To discretize the equation we consider the following difference approximations

.. math::
   p^{n+1}_{i,j} = 2 p^{n}_{i,j} - p^{n-1}_{i,j} + \Delta t^2 \left( D_{xx}p^{n} + D_{yy}p^{n} \right)

where

.. math::
   
  D_{xx} p^{n} = \frac{1}{\Delta x^2} \left( c_0 p^{n} + \sum_{k=1}^n c_k \left( p^{n}_{i+k,j} + p^{n}_{i-k,j} \right) \right), \\
  D_{yy} p^{n} = \frac{1}{\Delta y^2} \left( c_0 p^{n} + \sum_{k=1}^n c_k \left( p^{n}_{i,j+k} + p^{n}_{i,j-k} \right) \right) .

As in the previous example we consider the discretization on a structured grid. Here n corresponds to a time-step and (i,j)
corresponds to a location on the grid. 
   
------------
Gauss-Seidel
------------
In this example we revisit the equation solved by boundary value problem previously solved by the Jacobi method
and use a Red-Black Gauss-Seidel scheme. Traditionally, Gauss-Seidel scheme is inherently a serial algorithm but by
exploiting the structure of the problem we can color the domain in such a way to expose parallism. 



.. image:: figures/gsboard.png
   :scale: 10 %
   :align: center
