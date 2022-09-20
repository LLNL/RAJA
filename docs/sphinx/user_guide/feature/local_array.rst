.. ##
.. ## Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
.. ## and other RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _local_array-label:

===========
Local Array
===========

This section introduces RAJA *local arrays*. A ``RAJA::LocalArray`` is an
array object with one or more dimensions whose memory is allocated when a 
RAJA kernel is executed and only lives within the scope of the kernel 
execution. To motivate the concept and usage, consider a simple C++ example
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

Here, two stack-allocated arrays are defined inside the outer 'k' loop and 
used in both inner 'j' loops. This loop pattern may be also be expressed 
using RAJA local arrays in a ``RAJA::kernel_param`` kernel. We show a 
RAJA variant below, which matches the implementation above, and then discuss 
its constituent parts::

  // 
  // Define two local arrays
  // 

  using RAJA_a_array = RAJA::LocalArray<int, RAJA::Perm<0, 1>, RAJA::SizeList<5,7> >;
  RAJA_a_array kernel_a_array;

  using RAJA_b_array = RAJA::LocalArray<int, RAJA::Perm<0>, RAJA::SizeList<5> >;
  RAJA_b_array kernel_b_array;


  // 
  // Define the kernel execution policy
  // 

  using POL = RAJA::KernelPolicy<
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


  // 
  // Define the kernel
  // 

  RAJA::kernel_param<POL> ( RAJA::make_tuple(RAJA::RangeSegment(0,5), 
                                             RAJA::RangeSegment(0,7)),
                            RAJA::make_tuple(kernel_a_array, kernel_b_array),

    [=] (int j, int k, RAJA_a_array& kernel_a_array, RAJA_b_array& kernel_b_array) {
      a_array(k, j) = 5*k + j;
      b_array(j) = 5*k + j;
    },

    [=] (int j, int k, RAJA_a_array& a_array, RAJA_b_array& b_array) {
      printf("%d %d \n", kernel_a_array(k, j), kernel_b_array(j));
    }

  );

The RAJA version defines two ``RAJA::LocalArray`` types, one 
two-dimensional and one one-dimensional and creates an instance of each type. 
The template arguments for the ``RAJA::LocalArray`` types are:

  * Array data type
  * Index permutation (see :ref:`feat-view-label` for more on RAJA permutations)
  * Array dimensions

.. note:: ``RAJA::LocalArray`` types support arbitrary dimensions and sizes.

The kernel policy is a two-level nested loop policy (see 
:ref:`loop_elements-kernel-label` for information about RAJA kernel policies) 
with a statement type ``RAJA::statement::InitLocalMem`` inserted between the 
nested for-loops which allocates the memory for the local arrays when the 
kernel executes.  The ``InitLocalMem`` statement type uses a 'CPU tile' memory 
type, for the two entries '0' and '1' in the kernel parameter tuple 
(second argument to ``RAJA::kernel_param``). Then, the inner initialization 
loop and inner print loop are run with the respective lambda bodies defined 
in the kernel.

-------------------
Memory Policies
-------------------

``RAJA::LocalArray`` supports CPU stack-allocated memory and CUDA GPU shared
memory and thread private memory. See :ref:`localarraypolicy-label` for a
discussion of available memory policies.
