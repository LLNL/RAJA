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

.. _local_array-label:

===========
Local Array
===========

In this section we introduce RAJA local arrays, which enable developers to construct arrays between
subsequent loops. For clarity, the example below illustrates constructing an array between consecutive
loops in C++::

           for(int k = 0; k < 7; ++k) {

             int array[7][5];

             for(int j = 0; j < 5; ++j) {
               array[k][j] = 5*k +j;
             }

             for(int j = 0; j < 5; ++j) {
               printf("%d \n",array[k][j]);
             }

           }

Similary, RAJA offers constructs to intialize arrays between consecutive loops and have the array
accessible to subsequent loops. A RAJA local array is defined outside of a RAJA kernel by specifying
the type of the data and the dimensions of the array as template arguments.

In the example below we construct a two dimensional array of size 7 math:\times:5 ::

    using RAJA_array = RAJA::LocalArray<int, RAJA::Sizes<5,7> >;
    RAJA_array kernel_array;

.. note:: RAJA Local arrays support arbiratry dimensions and sizes.

Although the object has been constructed its memory has not yet been initalized.
Memory intialization for a RAJA local array is done through the ``InitLocalMem``
statement in a kernel policy ::

     RAJA::statement::InitLocalMem<array_policy, RAJA:ParamList<0>, statements...>

The InitLocalMemory statement is templated on an array policy which specifies where the RAJA local array
should be allocated and a RAJA::ParamList which identifies local arrays in the parameter list.
The current supported policies are listed below:

*  ``RAJA::cpu_tile_mem`` - Allocates memory on the stack
*  ``RAJA::cuda_shared_mem`` - Allocates memory in cuda shared memory
*  ``RAJA::cuda_thread_mem`` - Allocates memory in thread private memory

For completeness the RAJA analog of the loops above is illustrates below::

  using Pol =  RAJA::KernelPolicy<
                 RAJA::statement::For<1, RAJA::loop_exec,
                   RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::ParamList<0>,
                     RAJA::statement::For<0, RAJA::loop_exec,
                       RAJA::statement::Lambda<0>
                     >,
                    RAJA::statement::For<0, RAJA::loop_exec,
                      RAJA::statement::Lambda<1>
                    >
                  >
                >
              >;

    RAJA::kernel_param<Pol> (
        RAJA::make_tuple(RAJA::RangeSegment(0,5), RAJA::RangeSegment(0,7)),
        RAJA::make_tuple(kernel_array),

        [=] (int j, int k, RAJA_array &kernel_array) {
         kernel_array(k, j) = 5*k + j;
        },

        [=] (int j, int k, RAJA_array &kernel_array) {
         printf("%d \n", kernel_array(k, j));
       });