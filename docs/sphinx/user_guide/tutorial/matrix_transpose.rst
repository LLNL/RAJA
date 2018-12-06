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

------------------------------------
Matrix Transpose (Local Arrays)
------------------------------------

Key RAJA features shown in this example:

  * Basic usage of RAJA::kernel abstractions for nested loops
  * Multiple lambdas
  * RAJA local arrays

In this example, an input matrix A of dimension N x N is
reconfigured as a second matrix At with the rows of matrix
A reorganized as the columns of At and the columns
of matrix A as the rows of At.

This operation is carried out using a local memory tiling
algorithm. The algorithm first loads matrix entries into a
thread shared tile, a two-dimensional array, and then
reads from the tile swapping the row and column indices for
the output matrix.

The algorithm is expressed as a collection of `outer`
and `inner` for loops. Iterations of the inner loop will load/read
data into the tile; while outer loops will iterate over the number
of tiles needed to carry out the transposition. For simplicity we assume
the tile size divides the number of rows and columns of the matrix.

As a starting point, we choose the tile dimensions smaller than the
dimensions of the matrix.

.. literalinclude:: ../../../../examples/tut_matrix-transpose.cpp
                   :lines: 104

Next, we calculate the bounds of the inner (tile entries) and outer loops
(number of tiles needed to tranpose the matrix). 

.. literalinclude:: ../../../../examples/tut_matrix-transpose.cpp
                   :lines: 109-113

For clarity, a typical C-style sequential tiled matrix transpose may look
like this:

.. literalinclude:: ../../../../examples/tut_matrix-transpose.cpp
                   :lines: 129-160


^^^^^^^^^^^^^^^^^^^^^
RAJA::kernel Variants
^^^^^^^^^^^^^^^^^^^^^
Critical to the tiling agorithm above is the construction of the array constructed
in between subsequent loops. Similar to the C-style algorithm, RAJA offers constructs
to create an array in between subsequent loops and accessible to proceeding lambdas.
A ``RAJA::LocalArray`` is an object which is defined prior to a RAJA kernel but whose
memory is intialized as the kernel policy is traversed, we refer the reader 
to :ref:`local_array-label` for more details.

As before we construct range segments to establish the iteration spaces for the `outer`
and `inner` loops,

.. literalinclude:: ../../../../examples/tut_matrix-transpose.cpp
                   :lines: 173-176

and define the RAJA local array type and object that we will be using in our kernel,

.. literalinclude:: ../../../../examples/tut_matrix-transpose.cpp
                   :lines: 185,189

.. note:: Although the local array has been constructed above its memory has not yet been intialized.

Initialization for the array is specified by the ``RAJA::InitLocalMem<array_pol, RAJA::ParamList<...>, stmts..>``
statement. For a CPU implementation, we may choose ``RAJA::cpu_tile_mem`` which will allocate the array on the stack.
With a CUDA backend we may either allocate memory under the shared memory space or thread private memory space via 
the following policies ``RAJA::cuda_shared_mem`` or ``RAJA::cuda_thread_mem`` respectively. For completeness,
we present the RAJA analogue of the C-style loops below

.. literalinclude:: ../../../../examples/tut_matrix-transpose.cpp
                   :lines: 200-267

The file ``RAJA/examples/tut_matrix-multiply.cpp``
contains the complete working example code for the examples described in this
section along with OpenMP and CUDA variants.