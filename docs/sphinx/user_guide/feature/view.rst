.. ##
.. ## Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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
arrays. For example, a matrix :math:`A` of dimension :math:`N_r \times N_c` is
typically allocated as::

   double* A = new double [N_r * N_c];

Using a one-dimensional array makes it necessary to convert
two-dimensional indices (rows and columns of a matrix) to a one-dimensional
pointer offset index to access the corresponding array memory location. One 
could introduce a macro such as::

   #define A(r, c) A[c + N_c * r]

to access a matrix entry in row `r` and column `c`. However, this solution has
limitations; e.g., additional macro definitions are needed when adopting a 
different matrix data layout or when using other matrices. To facilitate
multi-dimensional indexing and different indexing layouts, RAJA provides 
``RAJA::View`` and ``RAJA::Layout`` classes.

----------
RAJA View
----------

A ``RAJA::View`` object wraps a pointer and enables various indexing schemes
based on the definition of a ``RAJA::Layout`` object. We can
create a ``RAJA::View`` for a matrix with dimensions :math:`N_r \times N_c` 
using a RAJA View and a default RAJA two-dimensional Layout as follows::

   double* A = new double [N_r * N_c];

   const int DIM = 2;
   RAJA::View<double, RAJA::Layout<DIM> > Aview(A, N_r, N_c);

The ``RAJA::View`` constructor takes a pointer to the matrix data and the 
extent of each matrix dimension as arguments. The template parameters to 
the ``RAJA::View`` type define the pointer type and the Layout type; here, 
the Layout just defines the number of index dimensions. Using the resulting 
view object, one may access matrix entries in a row-major fashion (the 
default RAJA layout) through the View parenthesis operator::

   // r - row index of a matrix
   // c - column index of a matrix
   // equivalent to indexing as A[c + r * N_c]
   Aview(r, c) = ...;

A ``RAJA::View`` can support any number of index dimensions::

   const int DIM = n+1;
   RAJA::View< double, RAJA::Layout<DIM> > Aview(A, N0, ..., Nn);

By default, entries corresponding to the right-most index are contiguous 
in memory; i.e., unit-stride access. Each other index is offset by the 
product of the extents of the dimensions to its right. For example, the loop::

   // iterate over index n and hold all other indices constant
   for (int in = 0; in < Nn; ++in) {
     Aview(i0, i1, ..., in) = ...
   }

accesses array entries with unit stride. The loop::

   // iterate over index j and hold all other indices constant
   for (int j = 0; j < Nj; ++j) {
     Aview(i0, i1, ..., j, ..., iN) = ...
   }

access array entries with stride N :subscript:`n` * N :subscript:`(n-1)` * ... * N :subscript:`(j+1)`.

------------
RAJA Layout
------------

``RAJA::Layout`` objects support other indexing patterns with different
striding orders, offsets, and permutations. In addition to layouts created
using the default Layout constructor, as shown above, RAJA provides other 
methods to generate layouts for different indexing patterns. We describe 
these next.

Permuted Layout
^^^^^^^^^^^^^^^^

The ``RAJA::make_permuted_layout`` method creates a ``RAJA::Layout`` object 
with permuted index strides. That is, the indices with shortest to 
longest stride are permuted. For example,::

  std::array< RAJA::idx_t, 3> perm {{1, 2, 0}};
  RAJA::Layout<3> layout = 
    RAJA::make_permuted_layout( {{5, 7, 11}}, perm );

creates a three-dimensional layout with index extents 5, 7, 11 with 
indices permuted so that the first index (index 0 - extent 5) has unit 
stride, the third index (index 2 - extent 11) has stride 5, and the 
second index (index 1 - extent 7) has stride 55 (= 5*11).

.. note:: If a permuted layout is created with the *identity permutation* 
          (e.g., {0,1,2}, the layout is the same as if it were created by 
          calling the Layout constructor directly with no permutation.

The first argument to ``RAJA::make_permuted_layout`` is a C++ array whose
entries define the extent of each index dimension. **The double braces are 
required to prevent compilation errors/warnings about issues trying to 
initialize a sub-object.** The second argument is the striding permutation.

In the next example, we create the same permuted layout, then create
a ``RAJA::View`` with it in a way that tells the View which index has 
unit stride::

  const int s0 = 5;  // extent of dimension 0
  const int s1 = 7;  // extent of dimension 1
  const int s2 = 11; // extent of dimension 2

  double* B = new double[s0 * s1 * s2];

  std::array< RAJA::idx_t, 3> perm {{1, 2, 0}};
  RAJA::Layout<3> layout = 
    RAJA::make_permuted_layout( {{s0, s1, s2}}, perm );

  // The Layout template parameters are dimension, 'linear index' type, 
  // and the index with unit stride
  RAJA::View<double, RAJA::Layout<3, RAJA::Index_type, 0> > Bview(B, layout);

  // Equivalent to indexing as: B[i + j * s0 * s2 + k * s0]
  Bview(i, j, k) = ...; 

.. note:: Telling a view which index has unit stride makes the 
          multi-dimensional index calculation more efficient by avoiding
          multiplication by '1' when it is unnecessary. **This must be done
          so that the layout permutation and unit-stride index specification
          are the same to prevent incorrect indexing.**

Offset Layout
^^^^^^^^^^^^^^^^

The ``RAJA::make_offset_layout`` method creates a ``RAJA::Layout`` object 
with offsets applied to the indices. For example,::

  double* C = new double[11]; 

  RAJA::Layout<1> layout = RAJA::make_offset_layout<1>({{-5}}, {{5}});

  RAJA::View<double, RAJA::Layout<1> > Cview(C, layout);

creates a one-dimensional view with a layout that allows one to index into
it using indices in :math:`[-5, 5]`. In other words, one can use the loop::

  for (int i = -5; i < 6; ++i) {
    CView(i) = ...;
  } 

to initialize the values of the array. Each 'i' loop index value is converted
to array offset access index by subtracting the lower offset to it; i.e., in 
the loop, each 'i' value has '-5' subtracted from it to properly access the
array entry.

The arguments to the ``RAJA::make_offset_layout`` method are C++ arrays that
hold the start and end values of the indices. RAJA offset layouts support
any number of dimensions; for example::

  RAJA::Layout<2> layout = RAJA::make_offset_layout<2>({{-1, -5}}, {{2, 5}});

defines a two-dimensional layout that enables one to index into a view using 
indices :math:`[-1, 2]` in the first dimension and indices :math:`[-5, 5]` in
the second dimension. As we remarked earlier, double braces are needed to 
prevent compilation errors/warnings about issues trying to initialize a 
sub-object.

Permuted Offset Layout
^^^^^^^^^^^^^^^^^^^^^^^^

The ``RAJA::make_permuted_offset_layout`` method creates a ``RAJA::Layout`` 
object with permutations and offsets applied to the indices. For example,::

  std::array< RAJA::idx_t, 2> perm {{1, 0}};
  RAJA::Layout<2> layout = 
    RAJA::make_permuted_offset_layout<2>( {{-1, -5}}, {{2, 5}}, perm ); 

Here, the two-dimensional index space is :math:`[-1, 2] \times [-5, 5]`, the
same as above. However, the index strides are permuted so that the first 
index (index 0) has unit stride and the second index (index 1) has stride 4, 
since the first index dimension has length 4.

Complete examples illustrating ``RAJA::Layouts`` and ``RAJA::Views``  may 
be found in the :ref:`offset-label` and :ref:`permuted-layout-label`
tutorial sections.

-------------------
RAJA Index Mapping
-------------------

``RAJA::Layout`` objects can also be used to map multi-dimensional indices 
to *linear indices* (i.e., pointer offsets) and vice versa. This
section describes basic Layout methods that are useful for converting between 
such indices. Here, we create a three-dimensional layout 
with dimension extents 5, 7, and 11 and illustrate mapping between a 
three-dimensional index space to a one-dimensional linear space::

   // Create a 5 x 7 x 11 three-dimensional layout object
   RAJA::Layout<3> layout(5, 7, 11);

   // Map from 3-D index (2, 3, 1) to the linear index
   // Note that there is no striding permutation, so rightmost is stride-1
   int lin = layout(2, 3, 1); // lin = 188 (= 1 + 3 * 11 + 2 * 11 * 7)

   // Map from linear index to 3-D index
   int i, j, k;
   layout.toIndices(lin, i, j, k); // i,j,k = {2, 3, 1}

``RAJA::Layout`` also supports *projections*, where one or more dimension
extent is zero. In this case, the linear index space is invariant for 
those multi-dimensional index entries; thus, the 'toIndicies(...)' method 
will always return zero for each dimension with zero extent. For example::

   // Create a layout with second dimension extent zero
   RAJA::Layout<3> layout(3, 0, 5);

   // The second (j) index is projected out
   int lin1 = layout(0, 10, 0);   // lin1 = 0
   int lin2 = layout(0, 5, 1);    // lin2 = 1

   // The inverse mapping always produces a 0 for j
   int i,j,k;
   layout.toIndices(lin2, i, j, k); // i,j,k = {0, 0, 1}
