.. ##
.. ## Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _tiledmatrixtranspose-label:

----------------------
Tiled Matrix Transpose
----------------------

Key RAJA features shown in this example are:

  * ``RAJA::kernel`` usage with multiple lambdas
  * ``RAJA::statement::Tile`` type

In this example, we compute the transpose of an input matrix 
:math:`A` of size :math:`N_r \times N_c` and store the result in a second 
matrix :math:`At` of size :math:`N_c \times N_r`.

We compute the matrix transpose using a tiling algorithm, which iterates 
over tiles of the matrix A and performs a transpose copy of a tile without 
storing the tile in another array. The algorithm is expressed as a collection 
of outer and inner loops. Iterations of the inner loop will transpose each tile,
while outer loops iterate over the tiles.

We start with a non-RAJA C++ implementation, where we choose tile
dimensions smaller than the matrix dimensions. Note that we do not assume 
that tiles divide evenly the number of rows and and columns of the matrix.
However, we do assume square tiles. First, we define matrix dimensions: 

.. literalinclude:: ../../../../examples/tut_tiled-matrix-transpose.cpp
   :start-after: // _tiled_mattranspose_dims_start
   :end-before: // _tiled_mattranspose_dims_end
   :language: C++

Then, we wrap the matrix data pointers in ``RAJA::View`` objects to  
simplify the multi-dimensional indexing:

.. literalinclude:: ../../../../examples/tut_tiled-matrix-transpose.cpp
   :start-after: // _tiled_mattranspose_views_start
   :end-before: // _tiled_mattranspose_views_end
   :language: C++

Then, the non-RAJA C++ implementation looks like this:

.. literalinclude:: ../../../../examples/tut_tiled-matrix-transpose.cpp
   :start-after: // _cstyle_tiled_mattranspose_start
   :end-before: // _cstyle_tiled_mattranspose_end
   :language: C++

Note that we need to include a bounds check in the code to avoid indexing out 
of bounds when the tile sizes do not divide the matrix dimensions evenly.

^^^^^^^^^^^^^^^^^^^^^
RAJA::kernel Variants
^^^^^^^^^^^^^^^^^^^^^

For ``RAJA::kernel`` variants, we use ``RAJA::statement::Tile`` types
for the outer loop tiling and ``RAJA::tile_fixed`` types to 
indicate the tile dimensions. The complete sequential RAJA variant is:

.. literalinclude:: ../../../../examples/tut_tiled-matrix-transpose.cpp
   :start-after: // _raja_tiled_mattranspose_start
   :end-before: // _raja_tiled_mattranspose_end
   :language: C++

The ``RAJA::statement::Tile`` types compute the number of tiles needed to 
iterate over all matrix entries in each dimension and generate iteration 
index bounds for each tile, which are used to generate loops for the inner  
``RAJA::statement::For`` types. Thus, the bounds checking logic in the 
non-RAJA variant is not needed. Note that the integer template parameters
to these statement types refer to the entries in the iteration space tuple
passed to the ``RAJA::kernel`` method.

The file ``RAJA/examples/tut_tiled-matrix-transpose.cpp`` contains the complete working example code for the examples described in this section, including
OpenMP, CUDA, and HIP variants.

A more advanced version using RAJA local arrays for CPU cache blocking and
using GPU shared memory is discussed in :ref:`matrixtransposelocalarray-label`.

