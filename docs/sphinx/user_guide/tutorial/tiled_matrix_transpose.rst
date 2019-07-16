.. ##
.. ## Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/COPYRIGHT file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _tiledmatrixtranspose-label:

----------------------
Tiled Matrix Transpose
----------------------

Key RAJA features shown in this example:

  * ``RAJA::kernel`` usage with multiple lambdas
  * ``RAJA::statement::Tile`` policy type

In this example, we compute the transpose of an input matrix 
:math:`A` of size :math:`N_r \times N_c` and store the result in a second 
matrix :math:`At` of size :math:`N_c \times N_r`.

We compute the matrix transpose using a tiling algorithm, which iterates 
over tiles of the matrix A and performs a transpose copy of a tile without 
explicitly storing the tile. The algorithm is expressed as a collection 
of outer and inner for-loops. Iterations of the inner loop will
transpose tile entries; while outer loops will iterate over
the tiles needed to compute the transpose. 

We start with a non-RAJA C++ implementation, where we choose tile
dimensions smaller than the matrix dimensions. Note that we do not assume 
that tiles divide evenly the number of rows and and columns of the matrix.
However, we do assume square tiles.

.. literalinclude:: ../../../../examples/tut_tiled-matrix-transpose.cpp
                   :lines: 67-68,88

Next, we calculate the number of tiles needed to carryout the transpose.

.. literalinclude:: ../../../../examples/tut_tiled-matrix-transpose.cpp
                   :lines: 91-92

Then, the C++ implementation may look like the following:

.. literalinclude:: ../../../../examples/tut_tiled-matrix-transpose.cpp
                   :lines: 110-132

Note that we include a bounds check in the code to avoid indexing out of
bounds when the tile sizes do not divide the matrix dimensions evenly.

^^^^^^^^^^^^^^^^^^^^^
RAJA::kernel Variant
^^^^^^^^^^^^^^^^^^^^^

For the ``RAJA::kernel`` variant, we use ``RAJA::statement::Tile`` types
for the outer loop tiling, with ``RAJA::statement::tile_fixed`` parameters
which identify the tile dimensions. The ``RAJA::statement::Tile`` types 
compute the number of tiles needed to iterate over all matrix entries in each
dimension and generate iteration index values within the bounds of the
associated iteration space. The complete sequential RAJA variant is given below:

.. literalinclude:: ../../../../examples/tut_tiled-matrix-transpose.cpp
                   :lines: 159-175

The file ``RAJA/examples/tut_tiled-matrix-transpose.cpp`` contains the complete working example code for the examples described in this section, including
OpenMP and CUDA variants.

A more advanced version using RAJA local arrays for CPU cache blocking and
using GPU shared memory is discussed in :ref:`matrixtransposelocalarray-label`.

