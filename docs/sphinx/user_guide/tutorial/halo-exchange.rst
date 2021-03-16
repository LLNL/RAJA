.. ##
.. ## Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/COPYRIGHT file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _halo_exchange-label:

------------------------------------
Halo Exchange (Workgroup Constructs)
------------------------------------

Key RAJA features shown in this example:

  * ``RAJA::WorkPool`` workgroup construct
  * ``RAJA::WorkGroup`` workgroup construct
  * ``RAJA::WorkSite`` workgroup construct
  * ``RAJA::RangeSegment`` iteration space construct
  * RAJA workgroup policies

In this example, we show how to use the RAJA workgroup constructs to implement
buffer packing and unpacking for data halo exchange on a computational grid,
a common MPI communication operation. This may not provide a performance gain
on a CPU system, but it can significantly speedup halo exchange on a GPU 
system compared to using ``RAJA::forall`` to run individual packing/unpacking
kernels.

.. note:: Using an abstraction layer over RAJA can make it easy to switch
          between using individual ``RAJA::forall`` loops or the RAJA workgroup
          constructs to implement halo exchange packing and unpacking at
          compile time or run time.

We start by setting the parameters for the halo exchange by using default
values or values provided via command line input. These parameters determine 
the size of the grid, the width of the halo, the number of grid variables 
and the number of cycles.

.. literalinclude:: ../../../../examples/tut_halo-exchange.cpp
   :start-after: _halo_exchange_input_params_start
   :end-before: _halo_exchange_input_params_end
   :language: C++

Next, we allocate the variables array (the memory manager in
the example uses CUDA Unified Memory if CUDA is enabled). These grid variables
are reset each cycle to allow checking the results of the packing and
unpacking.

.. literalinclude:: ../../../../examples/tut_halo-exchange.cpp
   :start-after: _halo_exchange_vars_allocate_start
   :end-before: _halo_exchange_vars_allocate_end
   :language: C++

We also allocate and initialize index lists of the grid elements to pack and
unpack:

.. literalinclude:: ../../../../examples/tut_halo-exchange.cpp
   :start-after: _halo_exchange_index_list_generate_start
   :end-before: _halo_exchange_index_list_generate_end
   :language: C++

All the code examples presented below copy the data packed from the grid 
interior: 

  +---+---+---+---+---+
  | 0 | 0 | 0 | 0 | 0 |
  +---+---+---+---+---+
  | 0 | 1 | 2 | 3 | 0 |
  +---+---+---+---+---+
  | 0 | 4 | 5 | 6 | 0 |
  +---+---+---+---+---+
  | 0 | 7 | 8 | 9 | 0 |
  +---+---+---+---+---+
  | 0 | 0 | 0 | 0 | 0 |
  +---+---+---+---+---+

into the adjacent halo cells:

  +---+---+---+---+---+
  | 1 | 1 | 2 | 3 | 3 |
  +---+---+---+---+---+
  | 1 | 1 | 2 | 3 | 3 |
  +---+---+---+---+---+
  | 4 | 4 | 5 | 6 | 6 |
  +---+---+---+---+---+
  | 7 | 7 | 8 | 9 | 9 |
  +---+---+---+---+---+
  | 7 | 7 | 8 | 9 | 9 |
  +---+---+---+---+---+


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Packing and Unpacking (Basic Loop Execution)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A sequential non-RAJA example of packing:

.. literalinclude:: ../../../../examples/tut_halo-exchange.cpp
   :start-after: _halo_exchange_sequential_cstyle_packing_start
   :end-before: _halo_exchange_sequential_cstyle_packing_end
   :language: C++

and unpacking:

.. literalinclude:: ../../../../examples/tut_halo-exchange.cpp
   :start-after: _halo_exchange_sequential_cstyle_unpacking_start
   :end-before: _halo_exchange_sequential_cstyle_unpacking_end
   :language: C++


^^^^^^^^^^^^^^^^^^^^^^^^^^
RAJA Variants using forall
^^^^^^^^^^^^^^^^^^^^^^^^^^

A sequential RAJA example uses this execution policy type:

.. literalinclude:: ../../../../examples/tut_halo-exchange.cpp
   :start-after: _halo_exchange_loop_forall_policies_start
   :end-before: _halo_exchange_loop_forall_policies_end
   :language: C++

to pack the grid variable data into a buffer:

.. literalinclude:: ../../../../examples/tut_halo-exchange.cpp
   :start-after: _halo_exchange_loop_forall_packing_start
   :end-before: _halo_exchange_loop_forall_packing_end
   :language: C++

and unpack the buffer data into the grid variable array:

.. literalinclude:: ../../../../examples/tut_halo-exchange.cpp
   :start-after: _halo_exchange_loop_forall_unpacking_start
   :end-before: _halo_exchange_loop_forall_unpacking_end
   :language: C++


For parallel multi-threading execution via OpenMP, the example can be run
by replacing the execution policy with:

.. literalinclude:: ../../../../examples/tut_halo-exchange.cpp
   :start-after: _halo_exchange_openmp_forall_policies_start
   :end-before: _halo_exchange_openmp_forall_policies_end
   :language: C++

Similarly, to run the loops in parallel on a CUDA GPU, we would use this 
policy:

.. literalinclude:: ../../../../examples/tut_halo-exchange.cpp
   :start-after: _halo_exchange_cuda_forall_policies_start
   :end-before: _halo_exchange_cuda_forall_policies_end
   :language: C++


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RAJA Variants using workgroup constructs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using the workgroup constructs in the example requires defining a few more
policies and types:

.. literalinclude:: ../../../../examples/tut_halo-exchange.cpp
   :start-after: _halo_exchange_loop_workgroup_policies_start
   :end-before: _halo_exchange_loop_workgroup_policies_end
   :language: C++

which are used in a slightly rearranged version of packing. See how the comment
indicating where a message could be sent has been moved down after the call to
run on the workgroup:

.. literalinclude:: ../../../../examples/tut_halo-exchange.cpp
   :start-after: _halo_exchange_loop_workgroup_packing_start
   :end-before: _halo_exchange_loop_workgroup_packing_end
   :language: C++

Similarly, in the unpacking we wait to receive all of the messages before
unpacking the data:

.. literalinclude:: ../../../../examples/tut_halo-exchange.cpp
   :start-after: _halo_exchange_loop_workgroup_unpacking_start
   :end-before: _halo_exchange_loop_workgroup_unpacking_end
   :language: C++

This reorganization has the downside of not overlapping the message sends with
packing and the message receives with unpacking.

For parallel multi-threading execution via OpenMP, the example using workgroup
can be run by replacing the policies and types with:

.. literalinclude:: ../../../../examples/tut_halo-exchange.cpp
   :start-after: _halo_exchange_openmp_workgroup_policies_start
   :end-before: _halo_exchange_openmp_workgroup_policies_end
   :language: C++

Similarly, to run the loops in parallel on a CUDA GPU use these policies and
types, taking note of the unordered work ordering policy that allows the
enqueued loops to all be run using a single CUDA kernel:

.. literalinclude:: ../../../../examples/tut_halo-exchange.cpp
   :start-after: _halo_exchange_cuda_workgroup_policies_start
   :end-before: _halo_exchange_cuda_workgroup_policies_end
   :language: C++

The packing is the same as the previous workgroup packing examples with the
exception of added synchronization after calling run and before sending the
messages. The previous CUDA example used forall to launch
``num_neighbors * num_vars`` CUDA kernels and performed ``num_neighbors``
synchronizations to send each message in turn. Here, the reorganization to pack
all messages before sending lets us use an unordered CUDA work ordering policy
in the workgroup constructs that reduces the number of CUDA kernel launches to
one. It also allows us to synchronize once before sending all of the messages:

.. literalinclude:: ../../../../examples/tut_halo-exchange.cpp
   :start-after: _halo_exchange_cuda_workgroup_packing_start
   :end-before: _halo_exchange_cuda_workgroup_packing_end
   :language: C++

After waiting to receive all of the messages we use workgroup constructs using
a CUDA unordered work ordering policy to unpack all of the messages using a
single kernel launch:

.. literalinclude:: ../../../../examples/tut_halo-exchange.cpp
   :start-after: _halo_exchange_cuda_workgroup_unpacking_start
   :end-before: _halo_exchange_cuda_workgroup_unpacking_end
   :language: C++

Note that the synchronization after unpacking is done to ensure that
``group_unpack`` and ``site_unpack`` survive until the unpacking loop has
finished executing.


The file ``RAJA/examples/tut_halo-exchange.cpp`` contains a complete
working example code, with OpenMP, CUDA, and HIP variants.
