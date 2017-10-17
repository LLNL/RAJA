.. _wave::

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

.. literalinclude:: ../../../examples/example-wave.cpp
                    :lines: 249-282

Notably, we have included the ``RAJA_HOST_DEVICE`` decorator in the lambda to create a portable kernel that may be executed on either
the CPU (with a variety of execution policies) the GPU (via the ``cuda_exec`` policy). 

A full working version ``example-wave.cpp`` may be found in the example folder.
