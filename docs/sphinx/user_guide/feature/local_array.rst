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
RAJA local arrays are multi-dimensional arrays which are intialized between loops in a kernel policy
and may only be used within a RAJA kernel. As a starting point, the following is a C++ analogue
illustrating the construction and usage of two arrays between loops::

           for(int k = 0; k < 7; ++k) {

            int a_array[7][5];
            int b_array[5];

             for(int j = 0; j < 5; ++j) {
               a_array[k][j] = 5*k + j;
               b_array[j] = 7*j + k;
             }

             for(int j = 0; j < 5; ++j) {
               printf("%d %d \n",a_array[k][j], b_array[j]);
             }

           }


Similary, RAJA offers constructs to intialize arrays between loops.
A RAJA local array is defined outside of a RAJA kernel by specifying
the type of the data the array will hold, the permutation of the indices
(see :ref:`_view-label` for more on permuted layouts), and the dimensions
of the array.

In the example below we construct two arrays. The first array has index
dimension 5, and 7 with the second index having unit stride. The second
array is one dimensional and has dimension 5.::

  using RAJA_a_array = RAJA::LocalArray<int, RAJA::Perm<0, 1>, RAJA::SizeList<5,7> >;
  RAJA_a_array kernel_a_array;

  using RAJA_b_array = RAJA::LocalArray<int, RAJA::Perm<0>, RAJA::SizeList<5> >;
  RAJA_b_array kernel_b_array;

.. note:: RAJA local arrays support arbiratry dimensions and sizes.


Although the objects have been constructed, memory has not yet been allocated.
Memory allocation for a RAJA local array occurs when the kernel policy
is executed. Memory intialization for a RAJA local array is carried out through
the ``InitLocalMem`` statement in a kernel policy. To illustrate the memory
allocation and usage of a RAJA local array we consider the RAJA variant of
the C++ loops above. The corresponding policy is provided below::

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

The InitLocalMemory statement is templated on an array policy which specifies where memory
will be allocated and a ``RAJA::ParamList`` which identifies local arrays in the parameter tuple.
The current supported policies are listed below:

*  ``RAJA::cpu_tile_mem`` - Allocates memory on the stack
*  ``RAJA::cuda_shared_mem`` - Allocates memory in cuda shared memory
*  ``RAJA::cuda_thread_mem`` - Allocates memory in cuda thread private memory

Lastly, the corresponding lambdas and RAJA kernel method is provided below::

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