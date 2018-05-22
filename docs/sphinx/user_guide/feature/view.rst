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

Matrix and tensor objects are most naturally expressed in
scientific computing applications as multi-dimensional arrays. However,
for efficiency in C and C++, they are usually allocated as one-dimensional
arrays. For example, a matrix :math:`A` of dimension :math:`M \times N` is 
typically allocated as::

   double *A = new double [M*N];

Then, one is left to convert between two-dimensional indices 
(rows and columns of matrix) to a one dimensional pointer-offset index 
(location in contigous memory). To simplify indexing, a C-style solution 
may be to introduce a macro::

   #define A(r, c) A[c + N * r]

The parenthesis operator is used to access a value in a particular row `r` and 
column `c` and the macro carries out the index conversion. Unfortunately,
this simple solution is fairly limited. Using macros like this with other
matrices or permutations of the data layout requires additional macros. 
To simplify multidimensional indexing, RAJA introduces ``RAJA::View`` 
and ``RAJA::Layout`` objects which enable indexing into data of any dimension 
and provides various data layout options.

---------
RAJA View
---------

The ``RAJA::View`` is a class templated on pointer type and a ``RAJA::Layout`` 
type. The ``RAJA::Layout`` specifies the number of dimensions in an array. 
Wrapping the pointer of the :math:`M \times N` matrix in a ``RAJA::View``
is accomplished in the following manner::

   double *A = new double [M*N];

   const int DIM = 2;   
   RAJA::View<double, RAJA::Layout<DIM> > Aview(A, M, N);

The ``RAJA::View`` constructor takes the pointer and the number of entries 
in each dimension. Then, one may access entries of the matrix in a row-major 
fashion::

     //r - row of a matrix
     //c - column of a matrix
     Aview(r,c);

The ``RAJA::View`` maps a multi-dimensional row and column indices to 
a linear index to access the proper location in memory. 
Additionally, a ``RAJA::Layout`` can be templated on argument type and 
dimension with unit stride::

    const int DIM = 2; 
    const int indexUnitStride = 0;
    RAJA::View<double, RAJA::Layout<DIM, int, indexUnitStride> > Aview(A, M, N);

This has the advantage of avoiding some artithmetic operations in the linear
index calculations.

In addition, the ``RAJA::View`` constructor is variadic and can support an 
arbitrary number of dimensions::

  const int DIM = Nn;
  RAJA::View< double, RAJA::Layout<DIM, int, 0> > Aview(A, N1, ..., Nn);

-----------
RAJA Layout
-----------

Previously, we specified the number of entries in each matrix dimension inside 
the ``RAJA::View`` constructor. Alternatively, we may use a 
``RAJA::Layout`` with the same information::

    const int DIM = 2; 
    const int indexUnitStride = 0;
    Layout<DIM> layout(M,N);
    RAJA::View<double, RAJA::Layout<DIM, int, indexUnitStride> > Aview(A, layout);

The capabilities of a ``RAJA::Layout`` extend beyond storing array 
dimensionality. Layouts may be defined to permute how data is accessed 
through the parenthesis operator and offset the enumeration of the entries 
of the array. The default striding has the first index (left-most) as the 
longest stride, and the last (right-most) index with unit stride. Permuting 
the ``RAJA::View`` striding order may be accomplished by the 
``RAJA::make_permuted_layout`` method. For example, to create a layout with 
dimension sizes 5, 7, 11 with default striding, one may either pass the sizes
to the Layout constructor or use the ``RAJA::make_permuted_layout`` method::

   // Create a 5 x 7 x 11 three-dimensional layout object
   Layout<3> layout(5, 7, 11);          

   // The above is equivalent to:
   Layout<3> layout = RAJA::make_permuted_layout({{5, 7, 11}}, RAJA::as_array<RAJA::Perm<0,1,2> >::get() );

The first argument in the ``RAJA::make_permuted_layout`` method stores an 
array with the dimensionality of each component, while the second argument 
``RAJA::as_array<RAJA::Perm<0,1,2>>::get()`` speficies arguments with the 
shortest stride to longest stride. Passing the Layout object to a View will 
identify `k` (argument 0) as the index with the longest stride, followed by 
`j` (argument 1) as the second longest, and `i` (argument 2) to have the 
shortest stride.::

  const int DIM = 3;        
  Layout<3> layout = RAJA::make_permuted_layout({{5, 7, 11}}, RAJA::as_array<RAJA::Perm<0,1,2> >::get() );
  RAJA::View< double, RAJA::Layout<DIM> > BView(B, layout); 
  Bview(k,j,i);

To permute the order of the striding, one may permute the arguments of 
``RAJA::Perm``. The following method call::

  Layout<3> layout2 = RAJA::make_permuted_layout({{5, 7, 11}}, RAJA::as_array<RAJA::Perm<1,2,0> >::get() );
  RAJA::View< double, RAJALayout<DIM> > B2View(B, layout2); 
  B2view(k,j,i);

returns a layout in which `k` (argument 0) has the shortest stride, followed by `j` (argument 2), and then `i` (argument 1) with the longest stride.
In addition to permuting how data is accessed in memory, layouts also support 
offsetting the enumeration of array entries. This is accomplished through 
the ``RAJA::make_offset_layout`` method. The following example will offset 
the indices of an array of length 11 by -5 and thus the values are enumerated 
from -5 to 5::

  RAJA::Layout<1> layout = RAJA::make_offset_layout<DIM>({{-5}}, {{5}});
  RAJA::View<int, RAJA::OffsetLayout<DIM>> Aview(A,layout);

Additionally, ``RAJA::make_offset_layout<DIM>`` supports arbitrary index offsetting. The following example offsets the start indices and end indicies for a 
two dimensional array::

  const int DIM = 2;              
  RAJA::OffsetLayout<DIM> layout = RAJA::make_offset_layout<DIM>({{-1,-5}}, {{2,5}});
  RAJA::View<int, RAJA::OffsetLayout<DIM>> Bview(B, layout);

Complete examples on using ``RAJA::Layouts`` and ``RAJA::Views``  may be found 
in :ref:`offset-label` and :ref:`permuted-layout-label` under the tutorial 
section.
