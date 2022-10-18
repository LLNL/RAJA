.. ##
.. ## Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
.. ## and other RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _feat-tiling-label:

===========
Loop Tiling
===========

In this section, we discuss RAJA statements that can be used to tile nested
loops. Typical loop tiling involves partitioning an iteration space into 
a collection of "tiles" and then iterating over tiles in outer loops and 
indices within each tile in inner loops. Many scientific computing algorithms 
can benefit from loop tiling due to more efficient cache usage on a CPU or
use of GPU shared memory.

For example, consider an operation performed using a C-style for-loop with 
a range of [0, 10)::

  for (int i=0; i<10; ++i) {
    // loop body using index 'i'
  }

This May be written as a loop nest that iterates over five tiles of size two::

  int numTiles = 5;
  int tileDim  = 2;
  for (int t=0; t<numTiles; ++t) {
    for (int j=0; j<tileDim; ++j) {
      int i = j + tileDim*t; // Calculate global index 'i'
      // loop body using index 'i'
    }
  }

Next, we show how loop tiling can be written using RAJA with variations that
use different ``RAJA::kernel`` execution policy statement types.

Here is a way to write the tiled loop kernel above using ``RAJA::kernel``::

   using KERNEL_EXEC_POL =
     RAJA::KernelPolicy<
       RAJA::statement::Tile<0, RAJA::tile_fixed<2>, RAJA::seq_exec,
         RAJA::statement::For<0, RAJA::seq_exec,
           RAJA::statement::Lambda<0>
         >
       >
     >;

   RAJA::kernel<KERNEL_EXEC_POL>(
     RAJA::make_tuple(RAJA::TypedRangeSegment<int>(0,10)), 
     [=] (int i) {
       // kernel body using index 'i'
     }
   );

In RAJA, the simplest way to tile an iteration space is to use
``RAJA::statement::Tile`` and ``RAJA::statement::For`` statement types. A
``RAJA::statement::Tile`` type is similar to a ``RAJA::statement::For`` type, 
but takes a tile size as the second template argument. The 
``RAJA::statement::Tile`` type generates the outer loop over tiles and 
the ``RAJA::statement::For`` type iterates over each tile.  Nested together, 
these statements will pass the global index ('i' in the example) to the 
lambda expression as (kernel body) in a non-tiled version above.

.. note:: When using ``RAJA::statement::Tile`` and ``RAJA::statement::For`` 
          types together to define a tiled loop structure, the integer passed 
          as the first template argument to each statement type must be the 
          same. This indicates that they both apply to the same iteration space
          in the space tuple passed to the ``RAJA::kernel`` method.

RAJA also provides alternative statements that provide the tile number and 
local tile index, if needed inside the kernel body, as shown below::

  using KERNEL_EXEC_POL2 =
    RAJA::KernelPolicy<
      RAJA::statement::TileTCount<0, RAJA::statement::Param<0>, 
                                  RAJA::tile_fixed<2>, RAJA::seq_exec,
        RAJA::statement::ForICount<0, RAJA::statement::Param<1>, 
                                   RAJA::seq_exec,
          RAJA::statement::Lambda<0>
        >
      >
    >;


  RAJA::kernel_param<KERNEL_EXEC_POL2>(
    RAJA::make_tuple(RAJA::TypedRangeSegment<int>(0,10)),
    RAJA::make_tuple((int)0, (int)0),
    [=](int i, int t, int j) {

      // i - global index
      // t - tile number
      // j - index within tile
      // Then, i = j + 2*t (2 is tile size)

    }
  );

The ``RAJA::statement::TileTCount`` type indicates that the tile number will 
be passed to the lambda expression and the ``RAJA::statement::ForICount`` type 
indicates that the local tile loop index will be passed to the lambda 
expression. Storage for these values is specified in the parameter tuple, the 
second argument passed to the ``RAJA::kernel_param`` method. The 
``RAJA::statement::Param<#>`` type appearing as the second 
template parameter for each statement type indicates which parameter tuple 
entry, the tile number or local tile loop index, is passed to the lambda and 
in which order. Here, the tile number is the second lambda argument (tuple 
parameter '0') and the local tile loop index is the third lambda argument 
(tuple parameter '1').

.. note:: The global loop indices always appear as the first lambda expression
          arguments. Then, the parameter tuples identified by the integers 
          in the ``RAJA::Param`` statement types given for the loop statement 
          types follow.
