.. ##
.. ## Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _permuted-layout-label:

---------------------------------------------
Permuted Layout: Batched Matrix-Multiply
---------------------------------------------

Key RAJA features shown in the following example:

  * ``RAJA::forall`` loop traversal template
  *  RAJA execution policies
  * ``RAJA::View`` multi-dimensional data access
  * ``RAJA::make_permuted_layout`` method to permute data ordering

This example performs batched matrix multiplication for a set of
:math:`3 \times 3` matrices using two different data layouts.

Matrices :math:`A` and :math:`B` are multiplied with the product stored in
matrix :math:`C`. The notation :math:`A^{e}_{rc}` indicates the row r and 
column c entry of matrix e. We describe the two data layouts we use for two
matrices. The extension to more than two matrices is straightforward. Using
different data layouts, we can assess which performs best for a given
execution policy and computing environment.

Layout 1:
Entries in each matrix are grouped together with each each having row major 
ordering; i.e.,

.. math::
  A = [A^{0}_{00}, A^{0}_{01}, A^{0}_{02},
       A^{0}_{10}, A^{0}_{11}, A^{0}_{12},
       A^{0}_{20}, A^{0}_{21}, A^{0}_{22},\\
       A^{1}_{00}, A^{1}_{01}, A^{1}_{02},
       A^{1}_{10}, A^{1}_{11}, A^{1}_{12},
       A^{1}_{20}, A^{1}_{21}, A^{1}_{22}];

Layout 2:
Matrix entries are first ordered by matrix index,
then by column index, and finally by row index; i.e.,

.. math::
  A = [A^{0}_{00}, A^{1}_{00}, A^{0}_{01},
       A^{1}_{01}, A^{0}_{02}, A^{1}_{02},
       A^{0}_{10}, A^{1}_{10}, A^{0}_{11},\\
       A^{1}_{11}, A^{0}_{12}, A^{1}_{12},
       A^{0}_{20}, A^{1}_{20}, A^{0}_{21},
       A^{1}_{21}, A^{0}_{22}, A^{1}_{22}];

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Permuted Layouts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next, we show how to construct the two data layouts using ``RAJA::View`` and
``RAJA::Layout`` objects. For more details on these RAJA concepts, please
refer to :ref:`view-label`.

The views for layout 1 are constructed as follows:

.. literalinclude:: ../../../../examples/tut_batched-matrix-multiply.cpp
   :start-after: _permutedlayout_defviews_start
   :end-before: _permutedlayout_defviews_end
   :language: C++

The first argument to ``RAJA::make_permuted_layout`` is a C++ array
whose entries correspond to the size of each array dimension; i.e., we have
'N' :math:`N_r \times N_c` matrices. The second argument describes the
striding order of the array dimensions. Note that since this case follows
the default RAJA ordering convention (see :ref:`view-label`), we use the 
identity permutation '(0,1,2)'. For each matrix, the column index (index 2) 
has unit stride and the row index (index 1) has stride 3 (number of columns). 
The matrix index (index 0) has stride 9 (:math:`N_c \times N_r`).

The views for layout 2 are constructed similarly:

.. literalinclude:: ../../../../examples/tut_batched-matrix-multiply.cpp
   :start-after: _permutedlayout_permviews_start
   :end-before: _permutedlayout_permviews_end
   :language: C++

Here, the first argument to ``RAJA::make_permuted_layout`` is the same as in
layout 1 since we have the same number of matrices, matrix dimensions and we
will use the same indexing scheme to access the matrix entries. However, the
permutation we use is '(1,2,0)'. This makes the matrix index (index 0) have 
unit stride, the column index (index 2) for each matrix has stride N, which 
is the number of matrices, and the row index (index 1) has 
stride :math:`N \times N_c`.

^^^^^^^^^^^^^^^^^^^
Example Code
^^^^^^^^^^^^^^^^^^^

Complete working examples that run the batched matrix-multiplication 
computation for both layouts and various RAJA execution policies is located
in the file ``RAJA/examples/tut_batched-matrix-multiply.cpp``. 

It compares the execution run times of the two layouts described above 
using four RAJA back-ends (Sequential, OpenMP, CUDA, and HIP). The OpenMP 
version for layout 1 looks like this:

.. literalinclude:: ../../../../examples/tut_batched-matrix-multiply.cpp
   :start-after: _permutedlayout_batchedmatmult_omp_start
   :end-before: _permutedlayout_batchedmatmult_omp_end
   :language: C++

The only differences between the lambda loop body for layout 1 and layout 2
cases are the names of the views. To make the algorithm code identical for all 
cases, we would use type aliases for the view and layout types in a header
file similarly to how we would abstract the execution policy out of the
algorithm.
