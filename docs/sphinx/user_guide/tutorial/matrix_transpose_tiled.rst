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

This section describes the implementation of a tiled matrix transpose kernel 
using both ``RAJA::kernel`` and ``RAJA::expt::launch`` interfaces. The intent
is to compare and contrast the two. The discussion builds on 
:ref:`matrixtranspose-label` by adding tiling to the matrix transpose 
implementation.

There are exercise files
``RAJA/exercises/kernel-matrix-transpose-tiled.cpp`` and
``RAJA/exercises/launch-matrix-transpose-tiled.cpp`` for you to work through 
if you wish to get some practice with RAJA. The files
``RAJA/exercises/kernel-matrix-transpose-tiled_solution.cpp`` and
``RAJA/exercises/launch-matrix-transpose-tiled_solution.cpp`` contain
complete working code for the examples. You can use the solution files to
check your work and for guidance if you get stuck.

Key RAJA features shown in this example are:

  * ``RAJA::kernel`` method and execution policies using the ``RAJA::statement::Tile`` type for loop tiling
  * ``RAJA::expt::launch`` kernel execution interface

In this example, we are still computing the transpose of an input matrix 
:math:`A` of size :math:`N_r \times N_c` and storing the result in a second 
matrix :math:`At` of size :math:`N_c \times N_r`.

We will compute the matrix transpose using a tiling algorithm, which iterates 
over tiles of the matrix A and performs a transpose operation on each tile.
The algorithm is expressed as a collection of outer and inner loops. 
Iterations of the inner loop will transpose each tile, while outer loops 
iterate over the tiles.

As in :ref:`matrixtranspose-label`, we start by defining the matrix dimensions.
Additionally, we define a tile size smaller than the matrix dimensions and 
determine the number of tiles in each dimension. Note that we do not assume 
that tiles divide evenly the number of rows and and columns of the matrix.
However, we do assume square tiles.

.. literalinclude:: ../../../../exercises/kernel-matrix-transpose-tiled_solution.cpp
   :start-after: // _tiled_mattranspose_dims_start
   :end-before: // _tiled_mattranspose_dims_end
   :language: C++

Then, we wrap the matrix data pointers in ``RAJA::View`` objects to  
simplify the multi-dimensional indexing:

.. literalinclude:: ../../../../exercises/kernel-matrix-transpose-tiled_solution.cpp
   :start-after: // _tiled_mattranspose_views_start
   :end-before: // _tiled_mattranspose_views_end
   :language: C++

Then, the C-style for-loop implementation looks like this:

.. literalinclude:: ../../../../exercises/kernel-matrix-transpose-tiled_solution.cpp
   :start-after: // _cstyle_tiled_mattranspose_start
   :end-before: // _cstyle_tiled_mattranspose_end
   :language: C++

Note that we need to include a bounds check in the code to avoid indexing out 
of bounds when the tile sizes do not divide the matrix dimensions evenly.

^^^^^^^^^^^^^^^^^^^^^^^^^^^
``RAJA::kernel`` Variants
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For ``RAJA::kernel`` variants, we use ``RAJA::statement::Tile`` types
for the outer loop tiling and ``RAJA::tile_fixed`` types to 
indicate the tile dimensions. The complete sequential RAJA variant is:

.. literalinclude:: ../../../../exercises/kernel-matrix-transpose-tiled_solution.cpp
   :start-after: // _raja_tiled_mattranspose_start
   :end-before: // _raja_tiled_mattranspose_end
   :language: C++

The ``RAJA::statement::Tile`` types compute the number of tiles needed to 
iterate over all matrix entries in each dimension and generate iteration 
index bounds for each tile, which are used to generate loops for the inner  
``RAJA::statement::For`` types. Thus, the explicit bounds checking logic in the 
C-style variant is not needed. Note that the integer template parameters
in the ``RAJA::statement::For`` types refer to the entries in the iteration 
space tuple passed to the ``RAJA::kernel`` method.

The ``RAJA::kernel`` CUDA variant is similar with the sequential execution
policies replaced with CUDA execution policies:

.. literalinclude:: ../../../../exercises/kernel-matrix-transpose-tiled_solution.cpp
   :start-after: // _raja_mattranspose_cuda_start
   :end-before: // _raja_mattranspose_cuda_end
   :language: C++

A notable difference between the CPU and GPU execution policy is the insertion
of the ``RAJA::statement::CudaKernel`` type in the GPU version, which indicates
that the execution will launch a CUDA device kernel.

The CUDA thread-block dimensions are set based on the tile dimensions and the
iterates withing each tile are mapped directly to GPU threads in each block
due to the ``RAJA::cuda_thread_{x, y}_direct`` policies.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``RAJA::expt::launch`` Variants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For ``RAJA::exp::launch`` variants, we use ``RAJA::expt::tile`` methods
for the outer loop tiling and ``RAJA::expt::loop`` methods
to iterate within the tiles. The complete sequential tiled 
``RAJA::expt::launch`` variant is:

.. literalinclude:: ../../../../exercises/launch-tiled-matrix-transpose_solution.cpp
   :start-after: // _raja_tiled_mattranspose_start
   :end-before: // _raja_tiled_mattranspose_end
   :language: C++

Similar to the ``RAJA::statement::Tile`` type in the ``RAJA::kernel`` variant
above, the ``RAJA::expt::tile`` method computes the number of tiles needed to 
iterate over all matrix entries in each dimension and generates a corresponding
iteration space for each tile, which is used to generate loops for the inner  
``RAJA::expt::loop`` methods. Thus, the explicit bounds checking logic in the 
C-style variant is not needed.

A CUDA ``RAJA::expt::launch`` tiled variant for the GPU is similar with 
different policies in the ``RAJA::expt::loop`` methods. The complete
``RAJA::expt::launch`` variant is:

.. literalinclude:: ../../../../exercises/launch-matrix-transpose_solution.cpp
   :start-after: // _raja_mattranspose_cuda_start
   :end-before: // _raja_mattranspose_cuda_end
   :language: C++

A notable difference between the CPU and GPU ``RAJA::expt::launch``
implementations is the definition of the compute grid. For the CPU
version, the argument list is empty for the ``RAJA::expt::Grid`` constructor.
For the CUDA GPU implementation, we define a 'Team' of one two-dimensional
thread-block with 16 x 16 = 256 threads.




