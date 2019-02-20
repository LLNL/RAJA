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

.. _matrixtransposelocalarray-label:

---------------------------------
Matrix Transpose with Local Array
---------------------------------

This section extends discussion in :ref:`tiledmatrixtranspose-label`, 
where only loop tiling is considered. Here, we combine loop tiling with 
``RAJA::LocalArray`` objects which enable CPU stack-allocated arrays, and
GPU thread local and shared memory to be used within kernels. For more 
information about ``RAJA::LocalArray``, please see :ref:`local_array-label`.

Key RAJA features shown in this example:

  * ``RAJA::kernel_param`` method with multiple lambda expressions
  * ``RAJA::statement::Tile`` type
  * ``RAJA::statement::ForICount`` type
  * ``RAJA::LocalArray``

As in :ref:`tiledmatrixtranspose-label`, this example computes the transpose 
of an input matrix :math:`A` of size :math:`N_r \times N_c` and stores the 
result in a second matrix :math:`At` of size :math:`N_c \times N_r`. The 
operation uses a local memory tiling algorithm. The algorithm tiles the outer 
loops and iterates over tiles in inner loops. The algorithm first loads 
input matrix entries into a local stack-allocated two-dimensional array for 
a tile, and then reads from the tile swapping the row and column 
indices to generate the output matrix. 

We start with a non-RAJA C++ implementation to show the algorithm pattern.
We choose tile dimensions smaller than the dimensions of the matrix and note 
that it is not necessary for the tile dimensions to divide evenly the number
of rows and columns in the matrix A.

.. literalinclude:: ../../../../examples/tut_matrix-transpose-local-array.cpp
                   :lines: 84-86,105

Next, we calculate the number of tiles needed to perform the transpose.

.. literalinclude:: ../../../../examples/tut_matrix-transpose-local-array.cpp
                   :lines: 108-109

The complete C++ implementation of the tiled transpose operation using 
local memory is:

.. literalinclude:: ../../../../examples/tut_matrix-transpose-local-array.cpp
                   :lines: 126-174

.. note:: * To prevent indexing out of bounds, when the tile dimensions do not
            divide evenly the matrix dimensions, we use a bounds check in the
            inner loops.
          * For efficiency, we order the inner loops so that reading from
            the input matrix and writing to the output matrix both use
            stride-1 data access.

^^^^^^^^^^^^^^^^^^^^^
RAJA::kernel Version
^^^^^^^^^^^^^^^^^^^^^

RAJA provides mechanisms to tile loops and use stack-allocated local arrays
in kernels so that algorithm patterns like we just described can be 
implemented with RAJA. A ``RAJA::LocalArray`` type specifies an object whose
memory is created inside a kernel using a ``RAJA::statement`` type in a RAJA 
kernel execution policy. The local array data is only usable within the kernel.
See :ref:`local_array-label` for more information. 

``RAJA::kernel`` methods also support loop tiling statements which determine 
the number of tiles needed to perform an operation based on tile size and
extent of the corresponding iteration space. Moreover, lambda expressions for
the kernel will not be invoked for iterations outside the bounds of an 
iteration space when tile dimensions do not divide evenly the size of the
iteration space; thus, no conditional checks on loop bounds are needed
inside inner loops.

For the RAJA version of the matrix transpose kernel above, we define the
type of the ``RAJA::LocalArray`` used for matrix entries in a tile: 

.. literalinclude:: ../../../../examples/tut_matrix-transpose-local-array.cpp
                   :lines: 192-193

The template parameters that define the type are: array data type, data stride
permutation for the array indices (here the identity permutation is given, so
the default RAJA conventions apply; i.e., the rightmost array index will be 
stride-1), and the array dimensions.

Here is the complete RAJA implementation for sequential CPU execution
with kernel execution policy and kernel:

.. literalinclude:: ../../../../examples/tut_matrix-transpose-local-array.cpp
                   :lines: 205-241

The ``RAJA::statement::Tile`` types at the start of the execution policy define
tiling of the outer 'row' (iteration space tuple index '1') and 'col' 
(iteration space tuple index '0') loops, including tile sizes 
(``RAJA::statement::tile_fixed`` types) and loop execution policies. Next, 
the ``RAJA::statement::InitLocalMem`` type initializes the local stack array
based on the memory policy type (``RAJA::cpu_tile_mem``). The 
``RAJA::ParamList<2>`` parameter indicates that the local array object is
associated with position '2' in the parameter tuple argument passed to the 
``RAJA::kernel_param`` method. Finally, we have two sets of nested inner loops
for reading the input matrix entries into the local array and writing them
out to the output matrix transpose. The inner bodies of each of these loop nests
are identified by lambda expression arguments '0' and '1', respectively.

A couple of notes about the nested inner loops are worth emphasizing. First, the
loops use ``RAJA::statement::ForICount`` types rather than 
``RAJA::statement::For`` types that we have seen in earlier ``RAJA::kernel``
nested loop examples. The ``RAJA::statement::ForICount`` type generates 
local tile indices that are passed to lambda loop body expressions. As 
the observant reader will observe, there is no local tile index computation 
needed in the lambdas for the RAJA version of the kernel as a result. The 
first integer template parameter for each ``RAJA::statement::ForICount`` type 
indicates the item in the iteration space tuple passed to the 
``RAJA::kernel_param`` method to which it applies; this is similar to 
``RAJA::statement::For`` usage. The second template parameter for each 
``RAJA::statement::ForICount`` type indicates the position in the parameter 
tuple passed to the ``RAJA::kernel_param`` method that will hold the 
associated local tile index. The loop execution policy template
argument that follows works the same as in ``RAJA::statement::For`` usage.
For more detailed discussion of RAJA loop tiling statement types, please see
:ref:`tiling-label`.

Now that we described the execution policy in some detail, let's pull 
everything together by briefly walking though the call to the 
``RAJA::kernel_param`` method. The first argument is a tuple of iteration
spaces that define the iteration pattern for each level in the loop nest.
Again, the first integer parameters given to the ``RAJA::statement::Tile`` and
``RAJA::statement::ForICount`` types identify the tuple entry they apply to.
The second argument is a tuple of data parameters that will hold the local
tile indices and ``RAJA::LocalArray`` tile memory. The tuple entries are 
associated with various statements in the execution policy as we described
earlier. Next, two lambda expression arguments are passed to the 
``RAJA::kernel_param`` method for reading and writing the input and output 
matrix entries, respectively.

Note that each lambda expression takes five arguments. The first two are
the matrix column and row indices associated with the iteration space tuple.
The next three arguments correspond to the parameter tuple entries. The first
two of these are the local tile indices used to access entries in the 
``RAJA::LocalArray`` object memory. The last argument is a reference to the 
``RAJA::LocalArray`` object itself.

The file ``RAJA/examples/tut_matrix-transpose-local-array.cpp`` contains the 
complete working example code for the examples described in this section along 
with OpenMP and CUDA variants.
