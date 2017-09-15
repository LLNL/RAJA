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

This is an overview of the examples included in RAJA and an overview view
of the different features each examples highlights.

---------------
Vector Addition
---------------
In this example, two arrays A, and B, of length N are added together
and the result is stored in a third vector C. As a starting point we begin
with a classic C++ style for loop and illustrate how to create a RAJA analog. 

.. code-block:: cpp
                
  for (int i = 0; i < N; ++i) {
    C[i] = A[i] + B[i];
  }

A RAJA analog of the previous example is simply the following

.. code-block:: cpp
                
  RAJA::forall<RAJA::exec_policy>(0, len, [=] (int i) {
    C[i] = A[i] + b[i];
  });

Where exec_policy may be any policy listed in the refence guide (TODO Add reference guide). 
  
---------------------
Matrix Multiplication
---------------------
In this example we multiply two matrices, A, and B, of dimension N X N
and store the result in a third marix C. We begin with a native C++ version
of matrix multiplication 

.. code-block:: cpp
                
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {

     double dot = 0.0;
      for (int k = 0; k < N; ++k){
        dot += Aview(row, k) * Bview(k, col);
      }
      
      Cview(row, col) = dot;    
    }
  }
                
and highlight the presence of nested loops. 



-------------
Jacobi Method
-------------
In this example we solve the following boundary value equation

.. math::
   
  U_{xx} + U_{yy} &= f, \quad U \in (0,1) \times (0,1), \\
  U(0,y) = U(1,y) &= 0, \\
  U(x,0) = U(x,1) &= 0.

Where

.. math::

  f = 2x(y-1)(y-2x+xy+2) e^{(x-y)}

To discretize the equation we consider a uniform grid
on the domain [0,1] \times [0,1]. Furthemore we approximate
spatial derivatives as

.. math::
   
   U_{xx} \approx \frac{U_{i+1,j} - 2U_{i,j} + U_{i+1,j}}{(\Delta x)^2}, \\
   U_{yy} \approx \frac{U_{i,j+1} - 2U_{i,j} + U_{i,j+1}}{(\Delta y)^2},

where (i,j) corresponds to a location on the structured grid. 

   
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

Here the equation is assumed to be discretized on a structured grid where n corresponds to a particular time step and (i,j)
corresponds to a location on the structured grid. 

   
---------------
Custom Indexset
---------------

---------------
Gauss-Sidel 
---------------
In this example we revisit the equation solved by the Jacobi method and consider
an alternative scheme, Gauss-Sidel. 
