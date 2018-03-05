.. ##
.. ## Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
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

Many scientific applications use multi-dimensional arrays for data structures
like matrices and tensors. In practice, large arrays are typically allocated 
on the heap by::

   double *A = new double [M*N]

Where A is a pointer to a contiguous array of data. If we want to access the
data entries in the array as a 2-dimensional M x N matrix, we could use macros::
  
   #define A(r, c) A[c + N * r]

By doing this, we are introducing a new **view** into the data **layout**.

---------
RAJA View
---------

To avoid macros and simplify multi-dimensional indexing, RAJA provides a 
``RAJA::View`` type, which wraps a pointer and overloads the parenthesis 
operator. Initializing a ``RAJA::View`` may be done as follows::

   RAJA::View< double, RAJA::Layout<n> > Aview(A, N1, ..., Nn);

The ``RAJA::View`` is templated on data type and layout dimension
(``RAJA::Layout<n>``). Here 'A' is a pointer to an array of doubles and
N1, ..., Nn are the sizes of each dimension. The arguments of the 
``RAJA::View`` are used to wrap the pointer and calculate the offset into 
the data for each access.

-----------
RAJA Layout
-----------

The ``RAJA::Layout<DIM>`` is particularly useful for creating multi-dimensional arrays with the ``RAJA::View`` class. Additionally, it may be used for mapping 
multi-dimensional indices to a one-dimensional indices and vice versa. In 
the following example, we consider a three-dimensional layout with dimension 
sizes 5, 7, and 11:: 

   // Create a 5 x 7 x 11 three-dimensional layout object
   Layout<3> layout(5, 7, 11);

Mapping from the three-dimensional index space to a one-dimensional linear 
space and vice versa is accomplished by::

   // Map from i=2, j=3, k=1 to the one-dimensional index
   int lin = layout(2,3,1); 

   // Map from linear space to 3d indices
   int i, j, k;
   layout.toIndices(lin, i, j, k); // i,j,k = {2, 3, 1}


The default striding has the first index (left-most) as the longest stride,
and the last (right-most) index with stride-1. Alternative layouts may be 
accomplished by using the ``RAJA::make_permuted_layout`` function. Basic usage
is as follows::

   // Create a layout object with the default striding order
   // The indices decrease in stride from left to right
   Layout<3> layout(5,7,11);

   // The above is equivalent to:
   Layout<3> default_layout = RAJA::make_permuted_layout({5,7,11}, PERM_IJK::value);
      
   // Create a layout object with permuted order
   // In this case, J is stride-1, and K has the longest stride
   Layout<3> perm_layout = RAJA::make_permuted_layout({5,7,11}, PERM_KIJ::value);
 
Permutation of up to rank 5 are provided with PERM_I ... PERM_IJKLM helper 
types.

Layout also supports projections, 0 or more dimensions may be of size zero.
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

An example that uses ``RAJA::View`` and ``RAJA::Layout`` types may be found
in the :ref:matrixmultiply-label` tutorial section.
