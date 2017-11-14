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

.. _view-label:

===============
View and Layout
===============

In machine learning and matrix algebra, multi-dimensional arrays are the work-horse data structure.
In practice large arrays are typically dynamically by ::

   double *A = new double [N1*N2]

Where conceptually A points to a rank two tensor of size :math:`Nx \times Ny \times Nz`. In standard C/C++
multidimensional indexing may be simplified through the use of macros ::

   #define A(x2, x1) A[x1 + N1 * x2]

In a more broader sense we are introducing a new **view** into the data's **layout**.

---------
RAJA View
---------
To bypass the need for macros and simplify multi-dimensional indexing, RAJA introduces the ``RAJA::View``. 
The ``RAJA::View`` wraps a pointer and overloads the paranthesis operator. The basic usage is as follows ::

   RAJA::View<dataType T, RAJA::Layout<DIM>> Aview(A, N1, ..., Nn);

The ``RAJA::View`` is templated on data type and layout (``RAJA::Layout``). The arguments for the instantization
of the view are the pointer to the data and the length of each dimension (i.e. ``N1, N2, ..., Nn``). 
Accessing the entries may then be done through the following accessor ::

   Aview(x2,x1)

-----------
RAJA Layout
-----------

The ``RAJA::Layout<DIM>`` is particularly useful for creating multi-dimensional arrays with the ``RAJA::View`` class, additionaly
it may be used for mapping multi-dimensional indecies to a one-dimensional indecies and vice versa. In the following example we consider 
creating a three-dimensional layout with dimension sizes 5, 7, and 11 :: 

   // Create a layout object
   Layout<3> layout(5,7,11);

Mappying from the three-dimensional index space to the linear space and vice versa is accomplished by:: 

   //Map from i=2, j=3, k=1 to the one-dimensional index
   int lin = layout(2,3,1); 

   // Map from linear space to 3d indices
   int i, j, k;
   layout.toIndices(lin, i, j, k); // i,j,k = {2, 3, 1}



The default striding has the first index (left-most) as the longest stride,
and the last (right-most) index with stride-1.
Layout supports projections, 0 or more dimensions may be of size zero.
In this case, the linear index space is invariant for those dimensions,
and toIndicies(...) will always produce a zero for that dimensions index.
An example of a "projected" Layout::
   
   // Create a layout with a degenerate dimensions
   Layout<3> layout(3, 0, 5);
 
   // The second (J) index is projected out
   int lin1 = layout(0, 10, 0);   // lin1 = 0
   int lin2 = layout(0, 5, 1);    // lin2 = 1

   // The inverse mapping always produces a 0 for J
   int i,j,k;
   layout.toIndices(lin2, i, j, k); // i,j,k = {0, 0, 1}


An application of the ``RAJA::View`` and ``RAJA::Layout<DIM>`` may be found in ``examples-matrix-multiply.cpp``.



