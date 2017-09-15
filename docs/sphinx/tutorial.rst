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
As a starting point we review key RAJA concepts starting with the
``RAJA::forall`` loop. 

.. code-block:: cpp
                
  RAJA::forall<exec_policy>(iter_space I, [=] (index_type i)) {
    //body
  });


1. [=] By-copy capture
2. [&] By-reference capture (for non-unified memory targets)
3. exec_policy - Specifies how the traversal occurs
4. iter_space  - Iteration space for RAJA loop (any random access container is expected)
5. index_type  - Index for RAJA loops
  

---------------
Vector Addition
---------------
In this example, two arrays A, and B, of length N are added together.
The result is stored in a third array C. As a starting point we begin
with a classic C++ style for loop and illustrate how to create a RAJA analog. 

.. code-block:: cpp
                
  for (int i = 0; i < N; ++i) {
    C[i] = A[i] + B[i];
  }

The RAJA analog is simply the following

.. code-block:: cpp
                
  RAJA::forall<RAJA::exec_policy>(0, N, [=] (int i) {
    C[i] = A[i] + b[i];
  });

where RAJA::exec_policy may be any policy listed in the refence guide.  
  
---------------------
Matrix Multiplication
---------------------
In this example we multiply two matrices, A, and B, of dimension N X N
and store the result in a third marix C. To simplify indexing we make use
of ``RAJA::Views``. A ``RAJA::View`` wraps a pointer to simplify
multi-dimensional indexing. The basic usage is as follows

.. code-block:: cpp
                
  double* A = new double[N*N];
  double* B = new double[N*N];
  double* C = new double[N*N];

  RAJA::View<double, RAJA::Layout<2>> Aview(A, N, N);
  RAJA::View<double, RAJA::Layout<2>> Bview(B, N, N);
  RAJA::View<double, RAJA::Layout<2>> Cview(C, N, N);

Where the arguments in ``RAJA::View`` denotes the type and layout of the data.
The argument in ``RAJA::Layout`` specifies the dimension of the data. In our case
we wish to treat the data as if it were two dimensional.

  

We begin with a native C++ version
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

With minimal disruption we can convert the outermost loop into a ``RAJA::forall`` loop.
Furthermore we will make use of the ``RAJA::RangeSegment`` enabling us to predifined loop bounds

.. code-block:: cpp
                
 RAJA::RangeSegment matBounds(0, N);

The resulting RAJA variant is as follows
 
.. code-block:: cpp
                
  RAJA::forall<exec_policy>(
    matBounds, [=](int row) {
  
      for (int col = 0; col < N; ++col) {

        double dot = 0.0;
        for (int k = 0; k < N; ++k) {
          dot += Aview(row, k) * Bview(k, col);
        }

        Cview(row, col) = dot;
        }
  });

In the case the user will not offload to a device ``RAJA::forall`` loops
may be nested.

.. code-block:: cpp

  RAJA::forall<RAJA::seq_exec>(
    matBounds, [=](int row) {  
      
    RAJA::forall<RAJA::seq_exec>(
      matBounds, [=](int col) {
          
      double dot = 0.0;
      for (int k = 0; k < N; ++k) {
        dot += Aview(row, k) * Bview(k, col);
      }
                
      Cview(row, col) = dot;
      });
  });
  

As general purpose nested loop, RAJA introduces the ``RAJA::forallN`` loop
which collapses a finite number of nested loops. This variant of the nested
loop may be used with any execution policy. Basic usage of the ``RAJA::forallN``
loop requires a ``RAJA::NestedPolicy<>`` and a ``RAJA::ExecList<>``,
which encapsulate how each loop of the should be traversed. 

.. code-block:: cpp

  RAJA::forallN<RAJA::NestedPolicy<
    RAJA::ExecList<RAJA::exec_policy, exec_policy>>>(
       matBounds, matBounds, [=](int row, int col) {
      
      double dot = 0.0;
      for (int k = 0; k < N; ++k) {        
        dot += Aview(row, k) * Bview(k, col);
      }
      
      Cview(row, col) = dot;
  });


-------------
Jacobi Method
-------------
In this example we solve the following boundary value equation

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
   
   U_{xx} \approx \frac{U_{i+1,j} - 2U_{i,j} + U_{i+1,j}}{(\Delta x)^2}, \\
   U_{yy} \approx \frac{U_{i,j+1} - 2U_{i,j} + U_{i,j+1}}{(\Delta y)^2},

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
   
---------------
Custom Indexset
---------------
This example illustrates how to construct a custom 
iteration space composed of segments. Here a segment
is an arbitrary collection of indices.
Assuming a grid with the following contents

(TODO - Add picture)

The following code will construct four segments wherein 
each segment will store indices corresponding to a particular
value on the grid. For example the first segment will store the
indices {0,2,8,10} corresponding to the location of values equal to 1.

---------------
Gauss-Seidel
---------------
In this example we revisit the equation solved by the Jacobi method consider the Gauss-Seidel scheme.
