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
arrays. For example, a matrix :math:`A` of dimension :math:`N_r \times N_c` is
typically allocated as::

   double* A = new double [N_r * N_c];

Using a one dimensional array makes it necessary to convert
two-dimensional indices (rows and columns of a matrix) to a one dimensional
pointer-offset index (location in contiguous memory). A C-style solution may 
be to introduce a macro::

   #define A(r, c) A[c + N_c * r]

The macro is used to access a matrix entry in a particular row, `r`, and
column, `c`, by performing the correct index conversion. Unfortunately,
this solution has serious limitations. For example, adopting a different 
matrix layout, or accessing different matrices in contiguous memory, would 
require additional macro definitions. To simplify multidimensional indexing 
and permutations, RAJA provides ``RAJA::View`` and ``RAJA::Layout`` classes 
which enable indexing into data using any number of dimensions as well as 
custom data layouts.

---------
RAJA View
---------

A ``RAJA::View`` object wraps a pointer and enables different indexing schemes
based on the type and definition of a ``RAJA::Layout`` object. Here, we 
create a ``RAJA::View`` for a matrix of dimensions :math:`N_r \times N_c` 
using a RAJA View and a simple RAJA Layout::

   double* A = new double [N_r * N_c];

   const int DIM = 2;
   RAJA::View<double, RAJA::Layout<DIM> > Aview(A, N_r, N_c);

The ``RAJA::View`` constructor takes a pointer and number of entries
in each dimension as its arguments. The template parameters to the 
``RAJA::View`` type define the pointer type and the Layout type, which 
defines the number of index dimensions. The resulting view object enables
data access in a row-major fashion through a parenthesis operator::

   // r - row of a matrix
   // c - column of a matrix
   // equivalent indexing as A[c + r*N_c]
   Aview(r,c) = ...;

A ``RAJA::View`` can support an arbitrary number of index dimensions::

   const int DIM = Nn+1;
   RAJA::View< double, RAJA::Layout<DIM> > Aview(A, N0, ..., Nn);

By default, entries in the right-most index are contiguous in memory; i.e.,
unit-stride access. Each entry to the left of that index is offset by the 
product of dimensions of the entries to its right. For example, the loop::

   // iterate over index n and hold all other indices constant
   for (int i = 0; i < Nn; ++i) {
     Aview(*, *, ..., i) = ...
   }

accesses array entries with unit stride. The loop::

   // iterate over index n-1 and hold all other indices constant
   for (int i = 0; i < N(n-1); ++i) {
     Aview(*, *, ..., i, *) = ...
   }

access array entries with stride :math:`Nn`, and so on.

------------
RAJA Layouts
------------

``RAJA::Layout`` objects support other indexing patterns such as different
striding orders, offsets, and permutations. In addition to layouts created
using the Layout constructor, as shown above, RAJA supports three other 
layouts for different indexing patterns. We describe these next.

Permuted Layouts
^^^^^^^^^^^^^^^^

The ``RAJA::make_permuted_layout`` method creates a ``RAJA::Layout`` object 
with permuted stridings; i.e., permute the indices with shortest to longest 
stride. For example,::

  RAJA::Layout<3> layout = 
    RAJA::make_permuted_layout({{5, 7, 11}}, 
                               RAJA::as_array< RAJA::Perm<1,2,0> >::get() );

creates a three-dimensional layout with index dimensions 5, 7, 11 with the
indices permuted so that the first index dimension (index 0 - size 5) has unit 
stride, the third index dimension (index 2 - size 11) has stride 5, and the 
second index dimension (index 1 - size 7) has stride 55 (= 5*11).

.. note:: If a permuted layout is created with the 'identity' permutation 
          (in this example RAJA::Perm<0,1,2>), the layout is the same as
          if it were created by calling the Layout constructor directly
          with no permutation.

The first argument to ``RAJA::make_permuted_layout`` is a C++ array whose
entries define the extent of each dimension. The double braces are required 
to prevent compilation errors/warnings about issues trying to initialize a 
sub-object. The second argument::

  RAJA::as_array< RAJA::Perm<0,1,2> >::get() 

takes a ``RAJA::Perm`` template argument that specifies the striding order of
the indices. The ``RAJA::as_array::get()`` method returns indices in the 
specified order. 

The next example creates the same permuted layout, then creates a ``RAJA::View``
with it in a way that tells the View which index has unit stride::

  const int s0 = 5;  // extent of dimension 0
  const int s1 = 7;  // extent of dimension 1
  const int s2 = 11; // extent of dimension 2

  double* B = new double[s0 * s1 * s2];

  RAJA::Layout<3> layout = 
    RAJA::make_permuted_layout({{s0, s1, s2}}, 
                               RAJA::as_array<RAJA::Perm<1,2,0> >::get() );

  // The Layout template parameters are dimension, index type, 
  // and the index with unit stride
  RAJA::View<double, RAJA::Layout<3, RAJA::Index_type, 1> > Bview(B, layout);

  // Equivalent to indexing as: B[i + j * s0 + k * s0 * s2]
  Bview(i, j, k) = ...; 

.. note:: Telling a view which index has unit stride makes the 
          multi-dimensional index calculation more efficient by avoiding
          multiplication by '1' when it is unnecessary. **This must be done
          with care (layout permutation and unit-stride index specification
          must be consistent) to prevent erroneous indexing.**

Offset Layouts
^^^^^^^^^^^^^^^^

The ``RAJA::make_offset_layout`` method creates a ``RAJA::Layout`` object 
with offsets applied to the indices. For example,::

  double* C = new double[11]; 

  RAJA::Layout<1> layout = RAJA::make_offset_layout<2>({{-5}}, {{5}});

  RAJA::View<double, RAJA::Layout<3> > Cview(C, layout);

creates a one-dimensional view with a layout that allows one to index into
it using the range :math:`[-5, 5]`. In other words, one can use the loop::

  for (int i = -5; i < 6; ++i) {
    CView(i) = ...;
  } 

to initialize the values of the array. Each 'i' loop index value is converted
to an actual data access index by adding 5 to it.

The arguments to the ``RAJA::make_offset_layout`` method are C++ arrays that
hold the start and end values of the indices. RAJA offset layouts support
any number of dimensions; for example::

  RAJA::Layout<2> layout = RAJA::make_offset_layout<2>({{-1,-5}}, {{2,5}});

defines a layout that enables one to index into a view using the range
:math:`[-1, -5] \times [2, 5]`. As we remarked earlier, double braces are
needed to prevent compilation errors/warnings about issues trying to 
initialize a sub-object.

Permuted Offset Layouts
^^^^^^^^^^^^^^^^^^^^^^^^

The ``RAJA::make_permuted_offset_layout`` method creates a ``RAJA::Layout`` 
object with permutations and offsets applied to the indices. For example,::

  RAJA::Layout<2> layout = 
    RAJA::make_permuted_offset_layout<2>({{-1,-5}}, {{2,5}}, 
                                         RAJA::as_array<RAJA::Perm<1, 0>>::get());

Here, the two-dimensional index range is :math:`[-1, -5] \times [2, 5]`, the
same as the previous example. Plus the index stridings are permuted so that
the first index (index 0) has unit stride and the second index (index 1) 
has stride 4, since the first index dimension has length 4.

Complete examples illustrating ``RAJA::Layouts`` and ``RAJA::Views``  may 
be found in the :ref:`offset-label` and :ref:`permuted-layout-label`
tutorial sections.

-------------------
RAJA Index Mappings
-------------------

``RAJA::Layout`` objects are used most often to map multi-dimensional indices 
to a one-dimensional indices (i.e., pointer offsets) and vice versa. This
section describes some Layout methods that some folks may find useful for
converting between such indices. Here, we create a three-dimensional layout 
with dimension sizes 5, 7, and 11 and illustrate mapping between a 
three-dimensional index space to a one-dimensional linear space::

   // Create a 5 x 7 x 11 three-dimensional layout object
   RAJA::Layout<3> layout(5, 7, 11);

   // Map from i=2, j=3, k=1 to the one-dimensional index
   int lin = layout(2, 3, 1); // lin = 188

   // Map from linear space to 3d indices
   int i, j, k;
   layout.toIndices(lin, i, j, k); // i,j,k = {2, 3, 1}

``RAJA::Layout`` also support projections; i.e., where zero or more dimensions 
may be of size zero. In this case, the linear index space is invariant for 
those dimensions, and toIndicies(...) will always produce a zero for that 
dimension's index. An example of a projected Layout::

   // Create a layout with second dimension size zero
   RAJA::Layout<3> layout(3, 0, 5);

   // The second (j) index is projected out
   int lin1 = layout(0, 10, 0);   // lin1 = 0
   int lin2 = layout(0, 5, 1);    // lin2 = 1

   // The inverse mapping always produces a 0 for j
   int i,j,k;
   layout.toIndices(lin2, i, j, k); // i,j,k = {0, 0, 1}
