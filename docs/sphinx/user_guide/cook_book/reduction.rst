.. ##
.. ## Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
.. ## and other RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _cook-book-reductions-label:

=======================
Cooking with Reductions
=======================

Please see the following section for more info on RAJA reductions:

 * :ref:`feat-reductions-label`.


----------------------------
Reductions with RAJA::forall
----------------------------

Here is the setup for a simple reduction example::

  const int N = 1000;

  int vec[N];

  for (int i = 0; i < N; ++i) {

    vec[i] = 1;

  }

Here a simple sum reduction is performed in a for loop::

  int vsum = 0;

  // Run a kernel using the reduction objects
  for (int i = 0; i < N; ++i) {

    vsum += vec[i];

  }

The results of these operations will yield the following values:

 * vsum == 1000

Here a simple sum reduction is performed using RAJA::

  using reduce_policy = RAJA::seq_reduce;
  // using reduce_policy = RAJA::omp_reduce;
  // using reduce_policy = RAJA::omp_target_reduce;
  // using reduce_policy = RAJA::cuda_reduce;
  // using reduce_policy = RAJA::hip_reduce;
  // using reduce_policy = RAJA::sycl_reduce;

  using exec_policy = RAJA::seq_exec;
  // using exec_policy = RAJA::omp_parallel_for_exec;
  // using exec_policy = RAJA::omp_target_parallel_for_exec<256>;
  // using exec_policy = RAJA::cuda_exec_rec_for_reduce<256>;
  // using exec_policy = RAJA::hip_exec_rec_for_reduce<256>;
  // using exec_policy = RAJA::sycl_exec<256>;

  RAJA::ReduceSum<reduce_policy, int> vsum(0);

  RAJA::forall<exec_policy>( RAJA::RangeSegment(0, N),
    [=](RAJA::Index_type i) {

    vsum += vec[i];

  });

The results of these operations will yield the following values:

 * vsum.get() == 1000
