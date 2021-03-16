.. ##
.. ## Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
.. ## and other RAJA project contributors. See the RAJA/COPYRIGHT file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _tiling-label:

===========
Loop Tiling
===========

In this section, we discuss RAJA statements that can be used to tile nested
for-loops. Typical loop tiling involves partitioning an iteration space into 
a collection of "tiles" and then iterating over tiles in outer loops and 
entries within each tile in inner loops. Many scientific computing algorithms 
can benefit from loop tiling due to more efficient cache usage on a CPU or
use of GPU shared memory.

For example, an operation performed using a for-loop with a range of [0, 10)::

  for (int i=0; i<10; ++i) {
    // loop body using index 'i'
  }

May be expressed as a loop nest that iterates over five tiles of size two::

  int numTiles = 5;
  int tileDim  = 2;
  for (int t=0; t<numTiles; ++t) {
    for (int j=0; j<tileDim; ++j) {
      int i = j + tileDim*t; // Calculate global index 'i'
      // loop body using index 'i'
    }
  }

Next, we show how this tiled loop can be represented using RAJA. Then, we
present variations on it that illustrate the usage of different RAJA kernel
statement types.

.. code-block:: cpp

   using KERNEL_EXEC_POL =
     RAJA::KernelPolicy<
       RAJA::statement::Tile<0, RAJA::tile_fixed<2>, RAJA::seq_exec,
         RAJA::statement::For<0, RAJA::seq_exec,
           RAJA::statement::Lambda<0>
         >
       >
     >;

   RAJA::kernel<KERNEL_EXEC_POL>(RAJA::make_tuple(RAJA::RangeSegment(0,10)), 
     [=] (int i) {
     // loop body using index 'i'
   });

In RAJA, the simplest way to tile an iteration space is to use RAJA 
``statement::Tile`` and ``statement::For`` statement types. A
``statement::Tile`` type is similar to a ``statement::For`` type, but takes
a tile size as the second template argument. The ``statement::Tile`` 
construct generates the outer loop over tiles and the ``statement::For`` 
statement iterates over each tile.  Nested together, as in the example, these 
statements will pass the global index 'i' to the loop body in the lambda 
expression as in the non-tiled version above.

.. note:: When using ``statement::Tile`` and ``statement::For`` types together
          to define a tiled loop structure, the integer passed as the first
          template argument to each statement type must be the same. This 
          indicates that they both apply to the same item in the iteration
          space tuple passed to the ``RAJA::kernel`` methods.

RAJA also provides alternative tiling and for statements that provide the tile 
number and local tile index, if needed inside the kernel body, as shown below::

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


  RAJA::kernel_param<KERNEL_EXEC_POL2>(RAJA::make_tuple(RAJA::RangeSegment(0,10)),
                                       RAJA::make_tuple((int)0, (int)0),
    [=](int i, int t, int j) {

      // i - global index
      // t - tile number
      // j - index within tile
      // Then, i = j + 2*t (2 is tile size)

   });

The ``statement::TileTCount`` type allows the tile number to be accessed as a
lambda argument and the ``statement::ForICount`` type allows the local tile 
loop index to be accessed as a lambda argument. These values are specified in 
the tuple, which is the second argument passed to the ``RAJA::kernel_param`` 
method above. The ``statement::Param<#>`` type appearing as the second 
template parameter for each statement type indicates which parameter tuple 
entry the tile number or local tile loop index is passed to the lambda, and 
in which order. Here, the tile number is the second lambda argument (tuple 
parameter '0') and the local tile loop index is the third lambda argument 
(tuple parameter '1').

.. note:: The global loop indices always appear as the first lambda expression
          arguments. Then, the parameter tuples identified by the integers 
          in the ``Param`` statement types given for the loop statement 
          types follow. 
