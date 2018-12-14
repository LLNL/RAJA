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

.. _matrixtranspose-label:

---------------------------------
Matrix Transpose with Local Array
---------------------------------

Key RAJA features shown in this example:

  * Basic usage of RAJA kernel
  * Multiple lambdas
  * Tile statement
  * ForICount statement
  * RAJA local arrays

In this example, an input matrix A of dimension N_r x N_c is
transposed and returned as a second matrix At of size N_c x N_r.

This operation is carried out using a local memory tiling
algorithm. The algorithm first loads matrix entries into an
iteration shared tile, a two-dimensional array, and then
reads from the tile swapping the row and column indices for
the output matrix.

The algorithm is expressed as a collection of `outer`
and `inner` for loops. Iterations of the inner loops will load
data into the tile; while outer loops will iterate over the number
of tiles needed to carry out the transpose.

Starting with a classic C++ implementation, we first choose tile 
dimensions smaller than the dimensions of the matrix. Furthermore, 
it is not necessary for the tile dimensions to divide the number
of rows and columns in the matrix A.

.. literalinclude:: ../../../../examples/tut_matrix-transpose.cpp
                   :lines: 84-85,105,108-109

Next, we calculate the number of tiles needed to carryout the transpose.

.. literalinclude:: ../../../../examples/tut_matrix-transpose.cpp
                   :lines: 108-109

Thus, the C++ implementation of a tiled transpose with local memory
may look like the following:

.. literalinclude:: ../../../../examples/tut_matrix-transpose.cpp
                   :lines: 126-167

.. note:: In the case the number of tiles leads to excess iterations, a bounds
          check is added to avoid indexing out of bounds. This occurs when the
          matrix dimensions are not divisible by the tile dimensions.

.. note:: For efficiency, we index into the column of the matrix using a unit
          stride. For this reason, the order of the second set of inner loops
          are swapped.


^^^^^^^^^^^^^^^^^^^^^
RAJA::kernel Variants
^^^^^^^^^^^^^^^^^^^^^
An important component of the algorithm above is the array used to store/read
entries of the matrix. Similar to the C++ algorithm, RAJA offers constructs
to create an array between loops. A RAJA::LocalArray is an object which is 
defined prior to a RAJA kernel but whose memory is initialized as the kernel policy
is executed. Furthermore, it may only be used within a RAJA kernel; we refer the
reader to :ref:`local_array-label` for more details. RAJA kernel also includes tiling 
statements which determine the number of necessary tiles to carry out the transpose 
and return iterate values within bounds of the original iteration space.

To construct the RAJA variant, we first construct a local array object:

.. literalinclude:: ../../../../examples/tut_matrix-transpose.cpp
                   :lines: 185-186

.. note:: Although the local array has been constructed, memory has not yet been allocated.

Tiling is then handled through RAJA's Tile statement. A caveat of solely using Tile is that 
only global indices are provided. To index into the local array this example employs
the ForICount statement which provides the iteration within the tile; we refer the reader
to :ref:`tiling-label` for more details. The complete sequential RAJA variant is given below:

.. literalinclude:: ../../../../examples/tut_matrix-transpose.cpp
                   :lines: 198-254

The file ``RAJA/examples/tut_matrix-transpose.cpp`` contains the complete working example code 
for the examples described in this section along with OpenMP and CUDA variants.