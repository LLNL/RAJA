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
                            
Ofcourse the standard C++ loop won't take advatage of the cores of multi/many-core processors,
but our RAJA analogue will! To construct the RAJA version we must first specify an execution policy
:ref:`ref-policy`
and construct an iteration space. 

                            
.. literalinclude:: ../../examples/example-add-vectors.cpp
                    :lines: 132-137

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

With minimal effort we can convert the outermost loop into a ``RAJA::forall`` loop.
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
   
---------------
Custom Indexset
---------------
This example illustrates how to construct a custom 
iteration space composed of segments. Here a segment
is an arbitrary collection of indices. In this example we wish
to create an iteration space composed of four segments coressponding
to the following grid

.. image:: figures/index_set_fig.png
   :scale: 40 %
   :align: center

Each segment will store incices corespoding to colors on the grid.
For example the first segment will store the indeces denoted by blue,
the second segment will store indeces denotes by red etc... 

In order to accomplish this we first create an instance of a
``RAJA::StaticIndexSet``

.. code-block:: cpp
                
   RAJA::StaticIndexSet<RAJA::TypedListSegment<RAJA::Index_type>> colorset;

In this example the StaticIndexSet is templated to hold TypedListSegments.

.. code-block:: cpp

  /*
    Buffer used for intermediate indices storage
  */
  auto *idx = new RAJA::Index_type[(n + 1) * (n + 1) / 4];

  /*
    Iterate over each dimension (DIM=2) for this example
  */

  for ( int xdim : {0,1}) {
    for ( int ydim : {0,1}) {
    
     RAJA::Index_type count = 0;

     
     /*
       Iterate over each dimension, incrementing by two to safely
       advance over neighbors
    */

    for (int xiter = xdim; xiter < n; xiter += 2) {
      for (int yiter = ydim; yiter < n; yiter += 2) {

      /*
        Add the computed index to the buffer
      */
      idx[count] = std::distance(std::addressof(Aview(0, 0)),
                                 std::addressof(Aview(xiter, yiter)));

      ++cout;
      }
    }

    /*
      RAJA::ListSegment - creates a list segment from a given array
      with a specific length.

      Here the indices are inserted from the buffer as a new ListSegment
    */
    colorset.push_back(RAJA::ListSegment(idx, count));
   }
  }

  delete[] idx;

Finally we have a custom colorset policy. With this policy we may have a ``RAJA::forall`` loop
transverse through each list segment stored in the colorset sequentially and transverse each
segment in parallel (if enabled). The policy may be defined as

.. code-block:: cpp
                
  using ColorPolicy = RAJA::ExecPolicy<RAJA::seq_segit, RAJA::omp_parallel_for_exec>;
  



---------------
Gauss-Seidel
---------------
In this example we revisit the equation solved by the Jacobi method consider the Gauss-Seidel scheme. Furthermore we build on the previous colorset example and 
