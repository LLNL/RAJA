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
As RAJA is built on the C++ lambda, we provide a brief overview on lambda functions.
Lambda's were introduced to allow for the construction of
in place functions. A lambda has the ability to "capture" variables from a local context and use within the
function.  A lambda expression takes the following form 

.. code-block:: cpp

   [capture list] (parameter list) {function body}  

The capture list corresponds to variables within the scope, while the parameter list corresponds to values which will be used within the function. By default, a lambda will capture by copying variables in the capture list. Capturing by reference may be accomplished by using the & symbol; for example 

.. code-block:: cpp

   int x;
   int y = 100;
   [&x, &y](){x=y;]

will generate a lambda which will assign the value of y to x. 
By setting the capture list as ``[=]`` or ``[&]`` all variables within scope will be captured by copy or reference respectively.

Building from the C++ lambda, RAJA introduces two types of templated methods, namely
``RAJA::forall`` and ``RAJA::forallN``. The ``RAJA::forall`` method is an abstraction
of the standard C++ loop which is templated on an execution policy, takes an iteration space, and a lambda capturing the loop body.

.. code-block:: cpp
                
  RAJA::forall<exec_policy>(iter_space I, [=] (index_type i)) {
    //body
  });

Similarly, the ``RAJA::ForallN`` loop is an abstraction of nested ``for`` loops. The ``RAJA::ForallN`` loop is
templated on up to N execution policies, expects an iteration space, and index for each execution policy.
  
.. code-block:: cpp
                
  RAJA::forallN<
    RAJA::NestedPolicy<exec_policy1, .... , exec_policyN> >(
      iter_space I1,..., iter_space IN, [=](index_type i1,..., index_type iN) {
         //body
  });

In summary, using one of the RAJA templated methods requires the developer to supply the following
1. Capture type [=] or [&]
3. exec_policy - Specifying how the traversal occurs
4. iter_space  - An iteration space for the RAJA loop (any random access container is expected)
5. index_type  - Index for RAJA loops
6. lambda      - capturing the body of the loop

The remainder of the tutorial demonstrates the utility of RAJA by drawing from commonly used
computing patterns.

---------------
Vector Addition
---------------
As a first example we consider vector addition. Here two vectors A, and B, of length N are added together
and the result is stored in a third vector, C. The C++ version of this loop takes the following form

.. literalinclude:: ../../examples/example-add-vectors.cpp
                    :lines: 93-95

The construction of a RAJA analog begins by first specifying an execution policy
(for more info see :ref:`ref-policy`) and constructing an iteration space. For this example we can generate
an iteration space composed of a contiguous sequence of numbers by using ``RAJA::RangeSegment`` (for more info see :ref:`ref-index`). 

.. literalinclude:: ../../examples/example-add-vectors.cpp
                    :lines: 107-112

By swapping out execution policies we can target different backends, the caveat being that the developer is reponsible for memory management (for more info see :ref:`ref-plugins`). Furthermore, using the cuda backend requires
the ``__device__`` decorator on the lambda. 

.. literalinclude:: ../../examples/example-add-vectors.cpp
                    :lines: 138-143

Lastly, invoking the CUDA execution policy requires specifying the number of threads per block.
A full working version ``example-add-vectors.cpp`` may be found in the example folder.

---------------------
Matrix Multiplication
---------------------
As a second example we consider matrix multiplication. Here two matrices, A, and B, of dimension
N x N are multiplied and the result is stored in a third matrix, C. 
Assuming that we have pointers to the data

.. literalinclude:: ../../examples/example-matrix-multiply.cpp
                    :lines: 107-109

and with the aid of macros

.. literalinclude:: ../../examples/example-matrix-multiply.cpp
                    :lines: 121-123

a C++ version of matrix multiplcation may be expressed as

.. literalinclude:: ../../examples/example-matrix-multiply.cpp
                    :lines: 136-146

With minimal effort we can create an RAJA analog by modifying the existing block of code.
First, we can relive the need of macros by making use of ``RAJA::View``, which
simplifies multi-dimensional indexing (for more info see :ref:`ref-view`). 
                           
.. literalinclude:: ../../examples/example-matrix-multiply.cpp
                    :lines: 155-157

Second, we can convert the outermost loop into a ``RAJA::forall`` loop

.. literalinclude:: ../../examples/example-matrix-multiply.cpp
                    :lines: 167-180

yielding code which may be excuted with various backends.
In the case that the code will not be off loaded to a device, ``RAJA::forall`` loops
may be nested

.. literalinclude:: ../../examples/example-matrix-multiply.cpp
                    :lines: 187-201  

Collapsing a series of nested loops may be done through the ``RAJA::forallN`` method.
Here it is necessary to use the ``RAJA::NestedPolicy`` (for more info see :ref:`ref-nested`) werein each
execution policy may be specified inside a ``RAJA::ExecList``. In the following example we pair the outer loop
with an OpenMP policy and the inner loop with a sequential policy.

.. literalinclude:: ../../examples/example-matrix-multiply.cpp
                    :lines: 229-240

A full working version ``example-matrix-multiply.cpp`` may be found in the example folder.

-------------
Jacobi Method
-------------

As a third example we consider solving the following boundary value problem

.. math::
   
  u_{xx} + u_{yy} &= f, \quad u \in (0,1) \times (0,1), \\
  u(0,y) = u(1,y) &= 0, \\
  u(x,0) = u(x,1) &= 0,

where

.. math::

  f = 2x(y-1)(y-2x+xy+2) e^{(x-y)} .

Here we choose the domain to be :math:`[-1,1] x [-1, 1]` and consider a
lattice discretization with equidistant nodes of spacing :math:`h`. 
The values we seek to approximate are the nodes inside the lattice, in order
to approximate the derivatives we consider the following discretization

.. math::
   
   u_{xx} \approx \frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{h^2}, \\
   u_{yy} \approx \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{h^2},

where :math:`(i,j)` corresponds to a location on lattice. After substituting the approximations
into the equation and some rearranging, the Jacobi method iterates

.. math::

   u^{k+1}_{ij} = \frac{1}{4} \left( -h^2 f_{ij} +  u^{k}_{ij} +  u^{k}_{ij} +  u^{k}_{ij} +  u^{k}_{ij} \right)

while

.. math::

   \mathcal{E}  = \sqrt{ \sum^{N}_{i} \sum^{N}_{j} \left( u_{ij}^{k+1} - u_{i,j}^{k} \right)^2 } > tol

In our example we take the intial guess :math:`\mathcal{u_{i,j}^0}` to be zero for all :math:`(i,j)`. As the solution to the equation is zero on the boundary we carry out each iteration on the interior nodes only. Pictorally, for each node on a grid (black) we carryout a weighted sum using four neighboring points (blue)

.. image:: figures/jacobi.png
   :scale: 10 %
   :align: center


Using the ``RAJA::ForallN`` method the iteration can be expressed as

.. literalinclude:: ../../examples/example-jacobi.cpp
                    :lines: 218-231

where we have predefined the ``jacobiompNestedPolicy`` as

.. literalinclude:: ../../examples/example-jacobi.cpp
                    :lines: 273-275

Computing :math:`\mathcal{E}` in parallel (aka reduction) requires multiple threads writting to the same location in memory, a procedure that is inherently not threadsafe. RAJA enables a thread-safe reduction by introducing ``RAJA::Reduce Sum`` variables. The following code illustrates carrying out the reduction procedure and updating the arrays used to hold the approximations

.. literalinclude:: ../../examples/example-jacobi.cpp
                    :lines: 298-305


A full working version ``example-jacobi.cpp`` may be found in the example folder.
           
-------------
Wave Equation
-------------
As an example of an evolution equation we consider the acoustic wave equation

.. math::  
   p_{tt} = c^{2} \left( p_{xx} + p_{yy} \right), \\
   (x,y) \in [-1,1] \times [-1,1]. 

As before we discretize the domain :math:`[-1,1] x [-1, 1]` by considering a lattice with equidistant nodes of distance :math:`h`. 
We take the discretization to be 

.. math::
   p^{n+1}_{i,j} = 2 p^{n}_{i,j} - p^{n-1}_{i,j} + \Delta t^2 \left( D_{xx}p^{n} + D_{yy}p^{n} \right)

where

.. math::
   
  D_{xx} p^{n} = \frac{1}{h^2} \left( c_0 p^{n} + \sum_{k=1}^N c_k \left( p^{n}_{i+k,j} + p^{n}_{i-k,j} \right) \right), \\
  D_{yy} p^{n} = \frac{1}{h^2} \left( c_0 p^{n} + \sum_{k=1}^N c_k \left( p^{n}_{i,j+k} + p^{n}_{i,j-k} \right) \right) .

The superscript :math:`n` denotes the :math:`n-th` timestep and the subscripts :math:`i,j` corresponds to a location on the grid.
In this example we choose the spatial discretization to be of fourth order (stencil width :math:`N=2`). The resulting kernel for the acoustic wave equation is given by

.. literalinclude:: ../../examples/example-wave.cpp
                    :lines: 249-282

Notably, we have included the ``RAJA_HOST_DEVICE`` decorator in the lambda to create a portable kernel that may be executed on either
the CPU (with a variety of execution policies) the GPU (via the ``cuda_exec`` policy). 

A full working version ``example-wave.cpp`` may be found in the example folder.

----------------------
Red-Black Gauss-Seidel
----------------------
In this example we revisit the boundary value problem solved by the Jacobi method and apply a Red-Black Gauss-Sidel scheme.
The classic Gauss-Sidel method is inherrently a sequential algorithm; however, parallism may be exposed by coloring the interior nodes
in the following manner

.. image:: figures/redblackGS.png
   :scale: 10 %
   :align: center

By applying such a coloring we may iterate over the different colors sequentially while updating each color in parallel. We can encapsulate the sequential/parallel transversal using by using a ``RAJA::NestedPolicy``.

.. literalinclude:: ../../examples/example-gauss-seidel.cpp
                    :lines: 240-241

Furthermore, we may construct a ``RAJA::StaticIndexSet`` which encapsulates the indices for the red and black nodes as two seperate list segments (for more info see ____ ). The following block of code illustrates the construction of the ``RAJA::StaticIndexSet``
           
.. literalinclude:: ../../examples/example-gauss-seidel.cpp
                    :lines: 188-220
                            
A full working version ``example-gauss-seidel.cpp`` may be found in the example folder.
