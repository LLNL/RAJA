.. ##
.. ## Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _permuted-layout-label:

-----------------------------------------------
Permuted Layout: Batched Matrix-Multiplication
-----------------------------------------------

This section contains an exercise file 
``RAJA/exercises/permuted-layout-batch-matrix-multiply.cpp`` for you to work 
through if you wish to get some practice with RAJA. The file 
``RAJA/exercises/permuted-layout-batch-matrix-multiply_solution.cpp`` contains 
complete working code for the examples discussed in this section. You can use 
the solution file to check your work and for guidance if you get stuck.
To build the exercises execute ``make permuted-layout-batch-matrix-multiply`` 
and ``make permuted-layout-batch-matrix-multiply_solution`` from the build 
directory.

Key RAJA features shown in the following example:

  * ``RAJA::forall`` loop traversal template
  *  RAJA execution policies
  * ``RAJA::View`` multi-dimensional data access
  * ``RAJA::make_permuted_layout`` method to permute data ordering

This example performs a "batched" matrix multiplication operation for a
collection of :math:`3 \times 3` matrices. Each pair of matrices 
:math:`A^{e}`, :math:`B^{e}`, indexed by 'e', is multiplied and the product 
is stored in a matrix :math:`C^{e}`. :math:`A^{e}` matrix entries, for all 
values of e, are stored in an array :math:`A`, all :math:`B^{e}` matrices 
are stored in array :math:`B`, and all :math:`C^{e}` matrices are stored in 
array :math:`C`. The notation :math:`A^{e}_{rc}` indicates the row r and 
column c entry of matrix e. 

We use two different data layouts for the arrays :math:`A`, :math:`B`, and 
:math:`C` to represent the storage pattern for the individual 
:math:`3 \times 3` matrices in each array. The following describes each layout
for two :math:`3 \times 3` matrices. The extension to more than two 
matrices is straightforward as you will see in the exercise code. Using 
different data layouts, we can assess which performs best for a given 
execution policy and computing environment. There may be additional performance
to be gained by additional tuning of memory layout for architecture and
execution back-end. But, this is beyond the scope of the exercise.

In **layout 1** entries for each matrix are contiguous in row major order; i.e.,
the ordering is column index, then row index, then matrix index:

.. math::
  A = [A^{0}_{00}, A^{0}_{01}, A^{0}_{02},
       A^{0}_{10}, A^{0}_{11}, A^{0}_{12},
       A^{0}_{20}, A^{0}_{21}, A^{0}_{22},\\
       A^{1}_{00}, A^{1}_{01}, A^{1}_{02},
       A^{1}_{10}, A^{1}_{11}, A^{1}_{12},
       A^{1}_{20}, A^{1}_{21}, A^{1}_{22}];

In **layout 2**, the matrix entries are first ordered by matrix index,
then by column index, and finally by row index:

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

.. literalinclude:: ../../../../exercises/permuted-layout-batch-matrix-multiply_solution.cpp
   :start-after: _permutedlayout_defviews_start
   :end-before: _permutedlayout_defviews_end
   :language: C++

The first argument to ``RAJA::make_permuted_layout`` is a C++ array
whose entries correspond to the extent of each layout dimension. Here, we have
:math:`N` :math:`N_r \times N_c` matrices. The second argument, the layout
permutation, describes the striding order of the array dimensions. Note that 
since this case follows the default RAJA ordering convention 
(see :ref:`view-label`), we use the identity permutation '(0,1,2)'. For each 
matrix, the column index (index 2) has unit stride and the row index (index 1) 
has stride :math:`N_c` (number of columns). The matrix index (index 0) has 
stride :math:`N_c \times N_r` (the number of entries in each matrix).

The views for layout 2 are constructed similarly:

.. literalinclude:: ../../../../exercises/permuted-layout-batch-matrix-multiply_solution.cpp
   :start-after: _permutedlayout_permviews_start
   :end-before: _permutedlayout_permviews_end
   :language: C++

Here, the first argument to ``RAJA::make_permuted_layout`` is the same as in
layout 1 since we have the same number of matrices with the same matrix 
dimensions, and we will use the same indexing scheme to access the matrix 
entries. However, the permutation we use is '(1,2,0)'. This makes the matrix 
index (index 0) have unit stride, the column index (index 2) for each matrix 
has stride N, which is the number of matrices, and the row index (index 1) has 
stride :math:`N \times N_c`.

^^^^^^^^^^^^^^^^^^^^^^
RAJA Kernel Variants
^^^^^^^^^^^^^^^^^^^^^^

The exercise files contain multiple RAJA variants of the batched matrix
multiplication kernel with different backends. They also print out execution
timings so you can compare the run times of the two layouts described above.
For example, the sequential CPU variant using layout 1 is:

.. literalinclude:: ../../../../exercises/permuted-layout-batch-matrix-multiply_solution.cpp
   :start-after: _permutedlayout_batchedmatmult_loop_start
   :end-before: _permutedlayout_batchedmatmult_loop_end
   :language: C++

The sequential CPU variant using layout 2 is:

.. literalinclude:: ../../../../exercises/permuted-layout-batch-matrix-multiply_solution.cpp
   :start-after: _permutedlayout2_batchedmatmult_loop_start
   :end-before: _permutedlayout2_batchedmatmult_loop_end
   :language: C++

The only differences between these two are the names of the views that appear
in the lambda expression loop body since a different layout is used to create 
view objects for each layout case. To make the algorithm code identical for all 
cases, we would use type aliases for the view and layout types in a header
file similarly to how we would abstract the execution policy out of the
algorithm, and compile the code for the case we want to run.

For comparison, here is an OpenMP CPU variant using layout 1:

.. literalinclude:: ../../../../exercises/permuted-layout-batch-matrix-multiply_solution.cpp
   :start-after: _permutedlayout_batchedmatmult_omp_start
   :end-before: _permutedlayout_batchedmatmult_omp_end
   :language: C++

The only difference between this variant and the sequential CPU variant shown
above is the exeution policy. The lambda expression loop body is identical
to the sequential CPU variant.

The exercise files also contain variants for RAJA CUDA and HIP back-ends.
Their similarities and differences are the same as what we've just described.
