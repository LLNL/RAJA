.. ##
.. ## Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _matrixtransposelocalarray-label:

---------------------------------
Matrix Transpose with Local Array
---------------------------------

This section extends the discussion in :ref:`tiledmatrixtranspose-label`, 
where only loop tiling is considered. Here, we combine loop tiling with 
``RAJA::LocalArray`` objects which enable us to store data for each tile in
CPU stack-allocated arrays or GPU thread local and shared memory to be used 
within kernels. For more information about ``RAJA::LocalArray``, please 
see :ref:`local_array-label`.

Key RAJA features shown in this example include:

  * ``RAJA::kernel_param`` method with multiple lambda expressions
  * ``RAJA::statement::Tile`` type
  * ``RAJA::statement::ForICount`` type
  * ``RAJA::LocalArray``
  * Specifying lambda arguments through statements

As in :ref:`tiledmatrixtranspose-label`, this example computes the transpose 
of an input matrix :math:`A` of size :math:`N_r \times N_c` and stores the 
result in a second matrix :math:`At` of size :math:`N_c \times N_r`. The 
operation uses a local memory tiling algorithm. The algorithm tiles the outer 
loops and iterates over tiles in inner loops. The algorithm first loads 
input matrix entries into a local two-dimensional array for a tile, and then 
reads from the tile swapping the row and column indices to generate the output 
matrix. 

We start with a non-RAJA C++ implementation to show the algorithm pattern.
We choose tile dimensions smaller than the dimensions of the matrix and note 
that it is not necessary for the tile dimensions to divide evenly the number
of rows and columns in the matrix A. As in the :ref:`tiledmatrixtranspose-label`
example, we start by defining the number of rows and columns in the matrices, 
the tile dimensions, and the number of tiles.

.. literalinclude:: ../../../../examples/tut_matrix-transpose-local-array.cpp
   :start-after: // _mattranspose_localarray_dims_start
   :end-before: // _mattranspose_localarray_dims_end
   :language: C++

We also use RAJA View objects to simplify the multi-dimensional indexing
as in the :ref:`tiledmatrixtranspose-label` example.

.. literalinclude:: ../../../../examples/tut_matrix-transpose-local-array.cpp
   :start-after: // _mattranspose_localarray_views_start
   :end-before: // _mattranspose_localarray_views_end
   :language: C++

The complete sequential C++ implementation of the tiled transpose operation 
using a stack-allocated local array for the tiles is:

.. literalinclude:: ../../../../examples/tut_matrix-transpose-local-array.cpp
   :start-after: // _mattranspose_localarray_cstyle_start
   :end-before: // _mattranspose_localarray_cstyle_end
   :language: C++

.. note:: * To prevent indexing out of bounds, when the tile dimensions do not
            divide evenly the matrix dimensions, we use a bounds check in the
            inner loops.
          * For efficiency, we order the inner loops so that reading from
            the input matrix and writing to the output matrix both use
            stride-1 data access.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RAJA::kernel Version of Tiled Loops with Local Array
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RAJA provides mechanisms to tile loops and use *local arrays*
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
type of the ``RAJA::LocalArray`` used for matrix entries in a tile and
create an object to represent it: 

.. literalinclude:: ../../../../examples/tut_matrix-transpose-local-array.cpp
   :start-after: // _mattranspose_localarray_start
   :end-before: // _mattranspose_localarray_end
   :language: C++

The template parameters that define the type are: array data type, data stride
permutation for the array indices (here the identity permutation is given, so
the default RAJA conventions apply; i.e., the rightmost array index will be 
stride-1), and the array dimensions. Next, we compare two RAJA implementations
of matrix transpose with RAJA. 

The complete RAJA sequential CPU variant with kernel execution policy and 
kernel is:

.. literalinclude:: ../../../../examples/tut_matrix-transpose-local-array.cpp
   :start-after: // _mattranspose_localarray_raja_start
   :end-before: // _mattranspose_localarray_raja_end
   :language: C++

The ``RAJA::statement::Tile`` types in the execution policy define
tiling of the outer 'row' (iteration space tuple index '1') and 'col' 
(iteration space tuple index '0') loops, including tile sizes 
(``RAJA::tile_fixed`` types) and loop execution policies. Next, 
the ``RAJA::statement::InitLocalMem`` type initializes the local stack array
based on the memory policy type (here, we use ``RAJA::cpu_tile_mem`` for
a CPU stack-allocated array). The ``RAJA::ParamList<2>`` parameter indicates 
that the local array object is associated with position '2' in the parameter 
tuple argument passed to the ``RAJA::kernel_param`` method. The first two
entries in the parameter tuple indicate storage for the local tile indices 
which can be used in multiple lambdas in the kernel. Finally, we have two sets 
of nested inner loops for reading the input matrix entries into the local 
array and writing them out to the output matrix transpose. The inner bodies of 
each of these loop nests are identified by lambda expression arguments 
'0' and '1', respectively.

Note that the loops over tiles use ``RAJA::statement::ForICount`` types 
rather than ``RAJA::statement::For`` types that we have seen in other
nested loop examples. The ``RAJA::statement::ForICount`` type generates 
local tile indices that are passed to lambda loop body expressions. As 
the reader will observe, there is no local tile index computation 
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

Now that we have described the execution policy in some detail, let's pull 
everything together by briefly walking though the call to the 
``RAJA::kernel_param`` method. The first argument is a tuple of iteration
spaces that define the iteration ranges for the level in the loop nest.
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

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RAJA::kernel Version of Tiled Loops with Local Array Specifying Lambda Arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The second RAJA variant works the same as the one above. The main differences
between the two variants is due to the fact that in this second one, we use 
``RAJA::statement::Lambda`` types to indicate which arguments each lambda 
takes and in which order. Here is the complete version including
execution policy and kernel:

.. literalinclude:: ../../../../examples/tut_matrix-transpose-local-array.cpp
   :start-after: // _mattranspose_localarray_raja_lambdaargs_start
   :end-before: // _mattranspose_localarray_raja_lambdaargs_end
   :language: C++

Here, the two ``RAJA::statement::Lambda`` types in the execution policy show
two different ways to specify the segments (``RAJA::Segs``) 
associated with the matrix column and row indices. That is, we can use a 
``Segs`` statement for each argument, or include multiple segment ids in one
statement. 

Note that we are using ``RAJA::statement::For`` types for the inner tile
loops instead of `RAJA::statement::ForICount`` types used in the first variant.
As a consequence of specifying lambda arguments, there are two main differences.
The local tile indices are properly computed and passed to the lambda 
expressions as a result of the ``RAJA::Offsets`` types that appear
in the lambda statement types. The ``RAJA::statement::Lambda`` type for each
lambda shows the two ways to specify the local tile index args; we can use an
``Offsets`` statement for each argument, or include multiple segment ids in one
statement. Lastly, there is only one entry in the parameter
tuple in this case, the local tile array. The placeholders are not needed.

The file ``RAJA/examples/tut_matrix-transpose-local-array.cpp`` contains the 
complete working example code for the examples described in this section along 
with OpenMP, CUDA, and HIP variants.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RAJA::expt::Launch Version of Tiled Loops with RAJA_TEAM_SHARED memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RAJA provides mechanisms to tile loops and use *local arrays*
in kernels so that algorithm patterns like we just described can be 
implemented with RAJA. Using the ``RAJA_TEAM_SHARED`` macro will create
GPU shared memory or memory on the stack when dispatching on the CPU.

``RAJA::expt::launch`` support methods for tiling over an iteration space
via ``RAJA::expt::tile``, and the ``RAJA::expt::loop_icount`` methods are
used to return the global iteration index and the local tile offset.
Moreover, lambda expressions for these methods will not be invoked for
iterations outside the bounds of an iteration space when tile dimensions
do not divide evenly the size of the iteration space; thus, no conditional
checks on loop bounds are needed inside inner loops.

The complete RAJA sequential CPU variant with kernel execution policy and 
kernel is:

.. literalinclude:: ../../../../exercises/launch_matrix-transpose-local-array_solution.cpp
   :start-after: // _mattranspose_localarray_raja_start
   :end-before: // _mattranspose_localarray_raja_end
   :language: C++

In this example ``RAJA::expt::tile methods`` are used to create tiles
of the outer 'row' and 'col' iteration spaces. The ``RAJA::expt::tile`` methods
take an additional argument specifying the tile size. To traverse the tile
we use the ``RAJA::expt::loop_icount`` methods, which are similar to ``RAJA::kernel``
ForICount statements. The ``RAJA::expt::loop_icount`` will generate the global and
local index with respect to the tile. The local tile index is necessary as we use it
to load entries from the global memory to ``RAJA_TEAM_SHARED`` memory.
