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

.. _permuted-layout-label:

---------------------------------------------
Batch Matrix-Multiply
---------------------------------------------

Key RAJA features shown in the following example:

  * ``RAJA::forall`` loop traversal template
  * RAJA execution policies
  * ``RAJA::View`` multi-dimensional data access
  * ``RAJA::make_permuted_layout`` permutes how data is accessed through the view parentheses operator

This example carries out batched matrix multiplication
for matrices of dimension 3 x 3 using two different
data layouts.

Matrices are stored in arrays A and B. Results
are stored in a third array, C.
The notation :math:`A^{e}_{rc}` is introduced
to correspond to the matrix entry in the row, r,
column, c, of matrix, e. Below we describe the two
layouts for the case of two (N=2) 3 x 3 matrices.

Layout 1:
Matrix entries are grouped together so that each
matrix is in a row major ordering, i.e.

.. math::
  A = [A^{0}_{00}, A^{0}_{01}, A^{0}_{02},
       A^{0}_{10}, A^{0}_{11}, A^{0}_{12},
       A^{0}_{20}, A^{0}_{21}, A^{0}_{22},\\
       A^{1}_{00}, A^{1}_{01}, A^{1}_{02},
       A^{1}_{10}, A^{1}_{11}, A^{1}_{12},
       A^{1}_{20}, A^{1}_{21}, A^{1}_{22}];

Layout 2:
Matrix entries are first ordered by matrix number,
then by column number, and finally by row number.

.. math::
  A = [A^{0}_{00}, A^{1}_{00}, A^{0}_{01},
       A^{1}_{01}, A^{0}_{02}, A^{1}_{02},
       A^{0}_{10}, A^{1}_{10}, A^{0}_{11},\\
       A^{1}_{11}, A^{0}_{12}, A^{1}_{12},
       A^{0}_{20}, A^{1}_{20}, A^{0}_{21},
       A^{1}_{21}, A^{0}_{22}, A^{1}_{22}];

The extension to N > 2 matrices follows by direct
extension. By exploring different data layouts,
we can assess which performs best under a given
execution policy and computing environment.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RAJA Permuted Layouts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The following code snippet constructs layout objects for the first data layout
we are considering.

.. literalinclude:: ../../../../examples/ex-batched-matrix-multiply.cpp
                    :lines: 97-100, 132-140

The first argument in the ``RAJA::make_permuted_layout`` is a C++ array
whose entries correspond to the dimensionality of each component.
The array is initialized using double braces as it enables initiation of the object
and its subobjects. The second argument is a ``RAJA::as_array`` object
templated on a RAJA permutation ``RAJA::Perm`` object. The template arguments
in ``RAJA::Perm``, :math:`0,1,2`, is used to to specify order of indices with the longest
to shortest stride, while ``RAJA::as_array::get()``
returns indices in the specified order. Indices are always enumerated with the left most as
:math:`0` and right most as the :math:`n^{th}` index. The example above uses the default
striding order in which the left most index (element number) has the longest stride and the right
most (column index) has unit stride. The following code example permutes the ordering
so that the element index (index :math:`0`) has unit stride, and the row index has the
longest stride (index :math:`1`).

.. literalinclude:: ../../../../examples/ex-batched-matrix-multiply.cpp
                    :lines: 149-157


We refer the reader to the :ref:`view-label` section
for a complete description of ``RAJA::View`` and ``RAJA::Layout`` objects.

^^^^^^^^^^^^^^^^^^^
RAJA Implementation
^^^^^^^^^^^^^^^^^^^
The complete example ``RAJA/examples/ex-offset.cpp`` compares batched matrix multiplication
using three different RAJA backends (Sequential, OpenMP, and CUDA) using the RAJA forall method.
Each version maintains the same loop body and compares run time with one of the possible layouts.
Additionally, timers are included in order to compare run time. The code example
below shows one of the RAJA variants:

.. literalinclude:: ../../../../examples/ex-batched-matrix-multiply.cpp
                    :lines: 193-226

As results will be platform specific, we invite readers to compare run-times in
their computing environments.
