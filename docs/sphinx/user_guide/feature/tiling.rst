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

In this section we discuss the RAJA tiling statements. In computing, a tiling
pattern is one in which an iteration space is broken up into a collection of
"tiles". An outer loop iterates through each tile and an inner loop
iterates through each entry of the tile. This type of pattern is advantageous
as operating on smaller local chunks of data can improve performance via
cache blocking. For example, a for loop with an iteration space of [0, 10)::

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

Tiling may be expressed in RAJA through RAJA kernel and tiling statements.
As a starting point, we provide a RAJA variant of the tiled C++ loops above::

    using KERNEL_EXEC_POL =
    RAJA::KernelPolicy<
      RAJA::statement::Tile<0, RAJA::statement::tile_fixed<2>, RAJA::seq_exec,
        RAJA::statement::For<0, RAJA::seq_exec,
                             RAJA::statement::Lambda<0>
        >
      >
    >;

   RAJA::kernel<KERNEL_EXEC_POL2>(RAJA::make_tuple(RAJA::RangeSegment(0,10)), [=] (int i) {

                         //body

   });

The ``Tile`` statement is similar to a RAJA for statement but with the added template argument of tile dimension.
The advantage here is that it is not necessary for the tile dimension to divide the iteration space. The policy 
will only return values within the range segment. The For statement in the policy specifies how the inner loop
should be executed. Both the Tile and For statements are templated on the same iteration space as they are derived from
the same range segment. This variant of tile statement will only return the global index to the lambda argument.
Next, we discuss variations on tile and for statements which provide the tile id number and iteration within the tile.

The statement ``TileTCount`` is used to extract the tile number, while the statement ``ForICount``
can be used to extract the iteration within the tile. TileTcount may be paired with a For statement,
and ForICount may be paired with the Tile statement. Unlike the Tile statement, return values 
for TileTCount and ForICount are returned in a kernel_param tuple and require specifying which tuple
entry should hold the iterate. For example, the following kernel policy will populate the tile number in
the first parameter entry and iteration within the tile in the second tuple entry. Parameter arguments 
follow segment arguments in lambda arguments and are returned after the global index. ::

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

