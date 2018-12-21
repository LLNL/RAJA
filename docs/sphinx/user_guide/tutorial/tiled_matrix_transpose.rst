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

.. _tiledmatrixtranspose-label:

----------------------
Tiled Matrix Transpose
----------------------

Key RAJA features shown in this example:

  * Basic usage of RAJA kernel
  * Multiple lambdas
  * Tile statement

In this example, an input matrix A of dimension N_r x N_c is
transposed and returned as a second matrix At of size N_c x N_r.

This operation is carried out using a tiling algorithm.
The algorithm iterates over tiles of the matrix A and
performs a transpose copy of a tile without explcitly 
storing the tile. A more advanced version using RAJA 
local arrays may be found here :ref:`matrixtransposelocalarray-label`.

The algorithm is expressed as a collection of ``outer``
and ``inner`` for loops. Iterations of the inner loop will
transpose tile entries; while outer loops will iterate over
the number of tiles needed to carryout the transposition.
We do not assume that tiles divide the number of rows and
and columns of the matrix.

Starting with a classic C++ implementation, we first choose tile
dimensions smaller than the dimensions of the matrix. Furthermore,
it is not necessary for the tile dimensions to divide the number
of rows and columns in the matrix A.

.. literalinclude:: ../../../../examples/tut_tiled-matrix-transpose.cpp
                   :lines: 75-76,96

Next, we calculate the number of tiles needed to carryout the transpose.

.. literalinclude:: ../../../../examples/tut_tiled-matrix-transpose.cpp
                   :lines: 99-100

Thus, the C++ implementation may look like the following:

.. literalinclude:: ../../../../examples/tut_tiled-matrix-transpose.cpp
                   :lines: 118-139

.. note:: In the case the number of tiles leads to excess iterations, a bounds
          check is added to avoid indexing out of bounds. Out of bounds indexing
          occurs when the matrix dimensions are not divisible by the tile dimensions.

^^^^^^^^^^^^^^^^^^^^^
RAJA::kernel Variants
^^^^^^^^^^^^^^^^^^^^^

To carryout the tiling pattern, RAJA kernel include tiling statements. The
``Tile`` statement determines the number of necessary tiles to carry 
out the transpose and return iterate values within bounds of the original 
iteration space. The Tile statement takes the dimension of the tile as a 
template parameter. The complete sequential RAJA variant is given below:

.. literalinclude:: ../../../../examples/tut_tiled-matrix-transpose.cpp
                   :lines: 166-185

The file ``RAJA/examples/tut_tiled-matrix-transpose.cpp`` contains the complete working example code
for the examples described in this section along with OpenMP and CUDA variants.