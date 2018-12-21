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

In this section we introduce RAJA local arrays.
A ``RAJA::LocalArray`` is a multi-dimensional object whose memory is allocated 
as a RAJA kernel policy is executed and may only be used inside kernel.
To motivate the concept and usage, we start by considering a simple C++ example
in which we construct and use two arrays in nested loops::

           for(int k = 0; k < 7; ++k) { //k loop

            int a_array[7][5];
            int b_array[5];

             for(int j = 0; j < 5; ++j) { //j loop
               a_array[k][j] = 5*k + j;
               b_array[j] = 7*j + k;
             }

             for(int j = 0; j < 5; ++j) { //j loop
               printf("%d %d \n",a_array[k][j], b_array[j]);
             }

           }

In the example above, we see that two arrays have been constructed inside the 
k - for loop and used in both j - for loops. This loop pattern may be expressed
similarly using RAJA local arrays in RAJA kernel. Next, we show the RAJA variant of
these for loops and discuss the different components:: 

  using RAJA_a_array = RAJA::LocalArray<int, RAJA::Perm<0, 1>, RAJA::SizeList<5,7> >;
  RAJA_a_array kernel_a_array;

  using RAJA_b_array = RAJA::LocalArray<int, RAJA::Perm<0>, RAJA::SizeList<5> >;
  RAJA_b_array kernel_b_array;

  using Pol =  RAJA::KernelPolicy<
                 RAJA::statement::For<1, RAJA::loop_exec,
                   RAJA::statement::InitLocalMem<RAJA::cpu_tile_mem, RAJA::ParamList<0, 1>,
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
  RAJA::make_tuple(kernel_a_array, kernel_b_array),

    [=] (int j, int k, RAJA_a_array &kernel_a_array, RAJA_b_array &kernel_b_array) {
      kernel_a_array(k, j) = 5*k + j;
      kernel_b_array(j) = 5*k + j;
    },

    [=] (int j, int k, RAJA_a_array &kernel_a_array, RAJA_b_array &kernel_b_array) {
      printf("%d %d \n", kernel_a_array(k, j), kernel_b_array(j));
    });

The RAJA implementation begins by first defining RAJA local array types and 
creating an instance of that type. RAJA local arrays are templated on: 

* Data type
* Index permutation ( see :ref:`view-label` for more on permuted layouts)
* Array dimensions

.. note:: RAJA local arrays support arbiratry dimensions and sizes.

Memory allocation for these objects occurs when the kernel policy is 
executed using the ``InitLocalMem`` statement. The InitLocalMem statement
is templated on a policy which specifies where memory for the local array
should be allocated while ``RAJA::ParamList`` identifies local arrays in 
the parameter tuple to intialize. The following policies may be used to specify memory allocation 
for the RAJA local arrays::

*  ``RAJA::cpu_tile_mem`` - Allocates memory on the stack
*  ``RAJA::cuda_shared_mem`` - Allocates memory in cuda shared memory
*  ``RAJA::cuda_thread_mem`` - Allocates memory in cuda thread private memory