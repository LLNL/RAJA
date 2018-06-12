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

Matrix and tensor objects are naturally expressed in
scientific computing applications as multi-dimensional arrays. However,
for efficiency in C and C++, they are usually allocated as one-dimensional
arrays. For example, a matrix, :math:`A`, of dimension :math:`N_r \times N_c` is
typically allocated as::

   double *A = new double [N_r * N_c];

Using a one dimensional array makes it necessary to convert
two-dimensional indices (rows and columns of a matrix) to a one dimensional
pointer-offset index (location in contiguous memory).
To simplify indexing, a C-style solution may be to introduce a macro::

   #define A(r, c) A[c + N_c * r]

The macro is used to access a matrix entry in a particular row, `r`, and
column, `c`, by performing the correct index conversion. Unfortunately,
this solution has serious limitations. For example, adopting a different matrix
layout, or accessing different matrices in contiguous memory, would require additional
macro definitions. To simplify multidimensional indexing and permutations,
RAJA provides ``RAJA::View`` and ``RAJA::Layout`` classes which enable
indexing into data of any number of dimensions as well as custom data layouts.

---------
RAJA View
---------
The ``RAJA::View`` is a class templated on pointer type and
a ``RAJA::Layout`` object. The following example creates a ``RAJA::View`` for
a matrix of dimensions :math:`N_r \times N_c`::

   double *A = new double [N_r * N_c];

   const int DIM = 2;
   RAJA::View<double, RAJA::Layout<DIM> > Aview(A, N_r, N_c);

The ``RAJA::View`` constructor takes a pointer and number of entries
in each dimension as its arguments. As a template argument, the ``RAJA::Layout`` object is used
to specify number of dimensions of the array. The resulting view object enables
data access in a row-major fashion through the parenthesis operator::

     //r - row of a matrix
     //c - column of a matrix
     //equivalent indexing as A[c + r*NCOLS]
     Aview(r,c)

Furthermore, the ``RAJA::View`` constructor is variadic and can support an
arbitrary number of dimensions::

  const int DIM = Nn+1;
  RAJA::View< double, RAJA::Layout<DIM> > Aview(A, N0, ..., Nn);

In the example above, we use the index subscript to enumerate the argument list.
By default, entries in the right most index are contiguous in memory.
Entries to the left of the :math:`n^{th}` index are offset by the product
of the dimensions of entries to the right. This offset is commonly referred to as
the stride, thus entries in the right most index have unit stride.

------------
RAJA Layout
------------
The capabilities of ``RAJA::Layout`` objects expand beyond storing array
dimensionality. A ``RAJA::Layout`` object may also be used as
an argument for a ``RAJA::View`` as well as to store dimension sizes,
striding order, and offsets for index enumeration. RAJA has four methods to create a layout object:

* ``RAJA::Layout<DIM> layout(N0, ..., Nn)``  Default constructor
* ``RAJA::make_permuted_layout`` Method which returns a layout with custom striding order
* ``RAJA::make_offset_layout``   Method which returns a layout with offset enumeration for each index
* ``RAJA::make_permuted_offset_layout`` Method which returns a layout with offset enumeration for each index and a custom striding order

At its core, a ``RAJA::Layout`` object is used for mapping multi-dimensional indices to
a one-dimensional index and vice versa. The default
constructor will create a layout object
in which the left-most index has the longest stride,
and the right-most index has unit stride. In the following example,
we consider a three-dimensional layout with dimension sizes 5, 7, and 11
and illustrate mapping between a three-dimensional index space to a one-dimensional linear
space::

   // Create a 5 x 7 x 11 three-dimensional layout object
   RAJA::Layout<3> layout(5, 7, 11);

   // Map from i=2, j=3, k=1 to the one-dimensional index
   int lin = layout(2,3,1); // lin = 188

   // Map from linear space to 3d indices
   int i, j, k;
   layout.toIndices(lin, i, j, k); // i,j,k = {2, 3, 1}

The ``RAJA::Layout`` object also support projections, i.e. zero or more dimensions may be of size zero.
In this case, the linear index space is invariant for those dimensions,
and toIndicies(...) will always produce a zero for that dimension's index.
An example of a projected Layout::

   // Create a layout with a degenerate dimensions
   RAJA::Layout<3> layout(3, 0, 5);

   // The second (j) index is projected out
   int lin1 = layout(0, 10, 0);   // lin1 = 0
   int lin2 = layout(0, 5, 1);    // lin2 = 1

   // The inverse mapping always produces a 0 for j
   int i,j,k;
   layout.toIndices(lin2, i, j, k); // i,j,k = {0, 0, 1}

By default, a ``RAJA::Layout`` object will covert between multi-dimensional
to one-dimensional indices by carrying out the scalar product between strides
and indices. This creates an unnecessary operation for the index of unit stride.
By using templating the ``RAJA::Layout`` object, developers can expose the dimension with unit stride
which enables the layout object to omit the extra calculation. This feature is
illustrated in the following section.

The ``RAJA::make_permuted_layout`` method enables the construction of ``RAJA::Layout``
objects with permuted striding orders. The example below, constructs a layout
with dimension sizes 5, 7, 11 with the left-most index having the
longest stride and the right-most index having unit stride::

  Layout<3> layout = RAJA::make_permuted_layout({{5, 7, 11}}, RAJA::as_array<RAJA::Perm<0,1,2> >::get() );

The first argument in the ``RAJA::make_permuted_layout`` is a C++ array
in which the entries correspond to the dimensionality of each component.
The array is initialized using double braces as it enables initiation of the object
and its subobjects. The second argument is a ``RAJA::as_array`` which is templated on
a ``RAJA::Perm`` object. The template arguments in ``RAJA::Perm``, :math:`0,1,2`,
are used to specify the striding order of the indices;
left most index corresponds to having the longest stride and the right most has unit stride.
The ``RAJA::as_array::get()`` method returns indices in the specified order. For clarity, the following example
illustrates using layout objects to define index striding order, expose index with unit
stride and template ``RAJA::View`` arguments::

  const int s0 = 5;  //stride of dimension 0
  const int s1 = 7;  //stride of dimension 1
  const int s2 = 11; //stride of dimension 2

  double *B = new double [s0 * s1 * s2];

  const int DIM = 3; //number of dimensions in array
  RAJA::Layout<DIM> layout = RAJA::make_permuted_layout({{s0, s1, s2}}, RAJA::as_array<RAJA::Perm<0,1,2> >::get() );

  //Layout is templated on dimensionality, index type, and index with unit stride
  RAJA::View<double, RAJA::Layout<DIM, RAJA::Index_type, 3> > Bview(B, layout);

  //Equivalent to indexing as
  //B[i + j * s2 + k * s2 * s1]
   Bview(k, j, i)

Templating on unit stride sets a stride associated with an index to one.
Thus templating an index with non-unit stride as unit stride may lead to undesired
indexing conversions. The following example illustrates the effects of marking an
index which natively does not have unit stride as unit stride::

  ...
  const int DIM = 3;
  Layout<DIM> layout = RAJA::make_permuted_layout({{s0, s1, s2}}, RAJA::as_array<RAJA::Perm<0,1,2> >::get() );

  //Layout is templated on dimensionality, index type, and index with unit stride
  RAJA::View<double, RAJA::Layout<DIM,RAJA::Index_type,0> > Bview(B, layout);

  //Equivalent to indexing as
  //B[i + j * s2 + k * 1]
   Bview(k, j, i)

As another example, reordering of the templated entries in ``RAJA::Perm``
will permute the striding so that index :math:`0` has the shortest stride,
while index :math:`1` will have the longest stride.::

  RAJA::Perm<1,2,0>

The example below illustrates basic usage and templates index :math:`0` as having unit stride::

  ...

  const int DIM = 3;
  Layout<DIM> layout = RAJA::make_permuted_layout({{s0, s1, s2}}, RAJA::as_array<RAJA::Perm<1,2,0> >::get() );
  //Layout is templated on dimensionality, index type, and index with unit stride
  RAJA::View<double, RAJA::Layout<DIM,RAJA::Index_type,0> > Bview(B, layout);

  //Equivalent to indexing as
  //B[k + s0 * i + j * s0 * s2]
  Bview(k, j, i)

The third approach to constructing a layout has the capability to offset index
enumerations. The following example uses the ``RAJA::make_offset_layout`` method
to offset the indices of an array of length :math:`11` by :math:`-5`. Thus, the array
entries are enumerated within the inclusive range of :math:`[-5, 5]`::

  RAJA::Layout<1> layout = RAJA::make_offset_layout<2>({{-5}}, {{5}});

The arguments for the ``RAJA::make_offset_layout`` method are standard C++ library
arrays which hold the desired start and end values of the index. As before double braces are
used to initialize the array and its subobjects. The ``RAJA::make_offset_layout`` supports offsetting
an arbitrary number of indices; the following example redefines the enumeration for a
two-dimensional array to be in the inclusive range of :math:`[-1, -5] \times [2, 5]`::

  RAJA::Layout<2> layout = RAJA::make_offset_layout<2>({{-1,-5}}, {{2,5}});

Lastly, the ``RAJA::make_offset_permuted_layout`` method pairs permuting stride ordering
with index offsets. The following example creates a layout object where the entries
in the left most index have the shortest stride and are enumerated between :math:`[-1,2]`
and entries in the right most index have the longest stride and are enumerated between :math:`[-5, 5]`::

  RAJA::Layout<2> layout = RAJA::make_offset_offset_layout<2>({{-1,-5}}, {{2,5}}, RAJA::as_array<RAJA::Perm<1, 0>>::get());

Complete examples on using ``RAJA::Layouts`` and ``RAJA::Views``  may be found
in :ref:`offset-label` and :ref:`permuted-layout-label` under the tutorial
section.
