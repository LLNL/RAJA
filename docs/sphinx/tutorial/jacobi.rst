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

.. _jacobi::

-----------
Jacobi
-----------

As a third example we consider solving the following boundary value problem

.. math::
   
  u_{xx} + u_{yy} &= f, \quad u \in (0,1) \times (0,1), \\
  u(0,y) = u(1,y) &= 0, \\
  u(x,0) = u(x,1) &= 0,

where

.. math::

  f = 2x(y-1)(y-2x+xy+2) e^{(x-y)} .


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Discretizing the equation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

.. image:: ../figures/jacobi.png
   :scale: 10 %
   :align: center

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2. RAJA ForallN method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using the ``RAJA::ForallN`` method the iteration can be expressed as

.. literalinclude:: ../../../examples/example-jacobi.cpp
                    :lines: 218-231

where we have predefined the ``jacobiompNestedPolicy`` as

.. literalinclude:: ../../../examples/example-jacobi.cpp
                    :lines: 273-275

Computing :math:`\mathcal{E}` in parallel (aka reduction) requires multiple threads writting to the same location in memory, a procedure that is inherently not threadsafe. RAJA enables a thread-safe reduction by introducing ``RAJA::Reduce Sum`` variables. The following code illustrates carrying out the reduction procedure and updating the arrays used to hold the approximations

.. literalinclude:: ../../../examples/example-jacobi.cpp
                    :lines: 298-305


A full working version ``example-jacobi.cpp`` may be found in the example folder.
