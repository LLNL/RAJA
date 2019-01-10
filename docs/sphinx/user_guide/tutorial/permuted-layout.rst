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

.. _permuted-layout-label:

---------------------------------------------
Batched Matrix-Multiply (Permuted Layouts)
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

Layout 1 is constructed as follows:

.. literalinclude:: ../../../../examples/tut_batched-matrix-multiply.cpp
                    :lines: 98-101, 134-143

The first argument to ``RAJA::make_permuted_layout`` is a C++ array
whose entries correspond to the size of each array dimension; i.e., we have
'N' :math:`N_r \times N_c` matrices. The second argument describes the
striding order of the array dimensions. Note that since this case follows
the default RAJA ordering convention (see :ref:`view-label`), we use the 
identity permutation '(0,1,2)'.

For each matrix, the column index (index 2) has unit stride and the row index
(index 1) has stride 3 (number of columns). The matrix index (index 0) has
stride 9 (:math:`N_c \times N_r`).

Layout 2 is constructed similarly:

.. literalinclude:: ../../../../examples/tut_batched-matrix-multiply.cpp
                    :lines: 157-163

Here, the first argument to ``RAJA::make_permuted_layout`` is the same as in
Layout 1 since we have the same number of matrices, matrix dimensions and we
will use the same indexing scheme to access the matrix entries. However, the
permutation we use is '(1,2,0)'.

This makes the matrix index (index 0) have unit stride, the column index 
(index 2) for each matrix has stride N, which is the number of matrices, and
the row index (index 1) has stride :math:`N \times N_c`.

^^^^^^^^^^^^^^^^^^^
Example Code
^^^^^^^^^^^^^^^^^^^

A complete working example that runs the batched matrix-multiplication 
computation for both layouts and various RAJA execution policies is located
in the file ``RAJA/examples/offset-layout.cpp``. It compares the execution run 
times of the two layouts using three RAJA back-ends (Sequential, OpenMP, and 
CUDA). The code example below shows the OpenMP version:

.. literalinclude:: ../../../../examples/tut_batched-matrix-multiply.cpp
                    :lines: 199-232

All versions use the exact same lambda loop body showing that data orderings
using RAJA can be altered similarly to execution policies without modifying
application source code directly.
