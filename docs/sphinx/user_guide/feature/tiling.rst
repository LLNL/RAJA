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

.. _tiling-label:

===========
Tiling
===========

In this section we discuss RAJA tiling statements. Many scientific
computation algorithms can be classified as "Tiling" algorithms. A Tiling
algorithm is one in which an iteration space is first broken up into a 
collection of "tiles" and nested loops are used to iterate over tiles and 
entries in a tile. This type of pattern is advantageous as operating on 
smaller local chunks of data can improve performance via cache blocking. 
For example, a for loop with a range of [0, 10)::

  for(int i=0; i<10; ++i) {
    //body
  }

May equivalently be expressed as loops which iterate over five tiles of 
dimension two::

      int numTiles = 5;
      int tileDim  = 2;
      for(int t=0; t<numTiles; ++t) {

        for(int j=0; j<tileDim; ++j) {

          //Calculation of global index
          int i = j + tileDim*t;
        }
      }

Tiling may be expressed in RAJA through kernel and tiling statements.
As a starting point, we provide a RAJA variant of the tiled C++ loops above::

    using KERNEL_EXEC_POL =
    RAJA::KernelPolicy<
      RAJA::statement::Tile<0, RAJA::statement::tile_fixed<2>, RAJA::seq_exec,
        RAJA::statement::For<0, RAJA::seq_exec,
                             RAJA::statement::Lambda<0>
        >
      >
    >;

   RAJA::kernel<KERNEL_EXEC_POL>(RAJA::make_tuple(RAJA::RangeSegment(0,10)), [=] (int i) {

                         //body

   });

To tile an iteration space, a developer must use ``Tile`` and ``For`` statements templated on the same
iteration space. The Tile statement is similar to a For statement but with the 
added template argument of tile dimension. The Tile statement will iterate over the necessary tiles
while the For statement will iterate over the local tile index. Together, the Tile and For statements will 
return the global index to the lambda argument. Next, we discuss variations on Tile 
and For statements which provide the tile number and local tile index.

The statement ``TileTCount`` is used to extract the tile number, while the statement ``ForICount``
can be used to extract the local tile index. Unlike the Tile and For statements, these statements 
require an additional template which indicate which param tuple entry holds the index. The ``RAJA::Param<#>``
statements are used to refer to which param tuple entry the For statement corresponds to.
For example, the following kernel policy will populate the tile number in the first param tuple entry and 
local tile index in the second param tuple entry. 


.. note :: The global index always appears as the first lambda argument. Param arguments then follow.

::

  using KERNEL_EXEC_POL2 =
    RAJA::KernelPolicy<
      RAJA::statement::TileTCount<0, RAJA::statement::Param<0>, RAJA::statement::tile_fixed<2>, RAJA::seq_exec,
        RAJA::statement::ForICount<0, RAJA::statement::Param<1>, RAJA::seq_exec,
          RAJA::statement::Lambda<0>
        >
      >
    >;


  RAJA::kernel_param<KERNEL_EXEC_POL2>(RAJA::make_tuple(RAJA::RangeSegment(0,10)),
                                 RAJA::make_tuple((int)0, (int)0),
                                 [=](int i, int t, int k) {

                                 //i - global index
                                 //t - tile number
                                 //k - iteration within tile
                                 });