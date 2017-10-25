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

.. _jacobi-label:

-----------------
Jacobi Iteration
-----------------

In this example, we solve the following boundary value problem using the
Jacobi iteration method.

.. math::
   
  u_{xx} + u_{yy} &= f, \quad u \in (0,1) \times (0,1), \\
  u(0,y) = u(1,y) &= 0, \\
  u(x,0) = u(x,1) &= 0,

where

.. math::

  f = 2x(y-1)(y-2x+xy+2) e^{(x-y)} .


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The discrete problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The problem domain in the example is the unit square :math:`[-1,1] x [-1, 1]`.
On the domain, we define grid with uniform mesh spacing :math:`h` in x and y.
The discrete solution values will be computed at the grid vertices that lie
at the intersection of the grid lines. Then, the approximate derivatives
are defined using finite differences as follows:

.. math::
   
   u_{xx} \approx \frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{h^2}, \\
   u_{yy} \approx \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{h^2},

where :math:`(i,j)` corresponds to grid vertex coordinate. After substituting 
the finite difference derivatives into the equation and some rearranging, 
the Jacobi method iterates the approximate solution 

.. math::

   u^{k+1}_{ij} = \frac{1}{4} \left( -h^2 f_{ij} +  u^{k}_{ij} +  u^{k}_{ij} +  u^{k}_{ij} +  u^{k}_{ij} \right)

until the residual is less than some tolerance and we consider the solution
'converged'

.. math::

   \mathcal{E}  = \sqrt{ \sum^{N}_{i} \sum^{N}_{j} \left( u_{ij}^{k+1} - u_{i,j}^{k} \right)^2 } < tol

We use and initial guess :math:`\mathcal{u_{i,j}^0} = 0` for all mesh points
:math:`(i,j)`. Since the Dirichlet boundary condition is zero at every point on
the domain boundary, we iterate the solution on the interior vertices only. 
At each iteration, the solution at each grid vertex is replaced by a weighted 
sum of the solution at four neighboring vertices (black vertex and blue 
neighboring vertices in the figure.

.. image:: ../figures/jacobi.png
   :scale: 10 %
   :align: center

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RAJA ForallN implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using the ``RAJA::ForallN`` template, the iteration can be expressed as

.. literalinclude:: ../../../examples/example-jacobi.cpp
                    :lines: 218-231

where we defined the ``jacobiompNestedPolicy`` type as

.. literalinclude:: ../../../examples/example-jacobi.cpp
                    :lines: 273-275

The template parameter after the nested policy specifies that an OpenMP 
parallel region is created around the loop nest. The nested policy indicates
that the loops are collapsed into one using the OpenMP *collapse* pragma and
a *nowait* pragma.

Note that computing :math:`\mathcal{E}` in parallel involves multiple threads 
writing to the same location in memory, which is inherently not thread-safe. 
RAJA provides thread-safe reduction types for all programming model back-ends.
The following code shows the loop that computes the residual using a RAJA 
OpenMP reduction type. 

.. literalinclude:: ../../../examples/example-jacobi.cpp
                    :lines: 298-305

The file ``RAJA/examples/example-jacobi.cpp``
contains the complete working example code.
