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

.. _reductions-label:

====================
Reductions
====================

RAJA does not provide a separate loop traversal template method for
reduction operations like some C++ template-based programming models do.
Instead RAJA provides reduction types that not only allow users to perform
reduction operations inside RAJA 'forall' loops in a portable, thread-safe
manner, but allow users to use as many reductions in a loop as they want.
Available reduction types in RAJA are described in this section.

----------------
Reduction Types
----------------

.. note:: * All RAJA reduction types are in the namespace ``RAJA``.
          * Each RAJA reduction type is templated on an *reduction policy*
            and a type for the reduction variable.

* ``ReducSum< reduce_policy, data_type >`` - Sum reduction.

* ``ReducMin< reduce_policy, data_type >`` - Min reduction.

* ``ReducMax< reduce_policy, data_type >`` - Max reduction.

* ``ReducMinLoc< reduce_policy, data_type >`` - Min reduction that provides the *first* loop iteration index where the minimum was found.

* ``ReducMaxLoc< reduce_policy, data_type >`` - Max reduction that provides the *first* loop iteration index where the maximum was found.

.. note:: When ``ReduceMinLoc`` and ``ReduceMaxLoc`` are used in a parallel
          execution context, the loop iteration index given for the min or max
          is not guaranteed to be the first in the iteration sequence. The
          index is determined by the ordering in which the reduction is
          finalized in parallel. The 'index' part of a "loc" reduction is 
          guaranteed to be reproducible for sequential execution only.

Here's a simple usage example::

  static constexpr int = 1000;

  //
  // Initialize array of length N with all ones. Then, set some other
  // values to make the example mildly interesting...
  //
  int vec[N] = {1};
  vec[100] = -10; vec[500] = -10;
  vec[5] = 10; vec[995] = 10;

  // Create sum and max reduction objects with initial values
  RAJA::ReduceSum< RAJA::omp_reduce, int > vsum(0);
  RAJA::ReduceMax< RAJA::omp_reduce, int > vmax(100);

  // Create 'loc' reduction objects with initial values and indices
  RAJA::ReduceMinLoc< RAJA::omp_reduce, int > vminloc(100, -1);
  RAJA::ReduceMaxLoc< RAJA::omp_reduce, int > vmaxloc(-100, -1);

  RAJA::forall<RAJA::omp_parallel_for_exec>( RAJA::RangeSegment(0, N),
    [=](Index_type i) {

    vsum += vec[i] ;
    vminloc.minloc( vec[i] ) ;
    vmaxloc.maxloc( vec[i] ) ;
    vmax.max( vec[i] ) ;

  });

  int my_vsum = static_cast<int>(vsum.get());
  int my_vmax = static_cast<int>(vmax.get());

  int my_vminloc = static_cast<int>(vminloc.get()));
  int my_vminidx = static_cast<int>(vminloc.getLoc()));

  int my_vmaxloc = static_cast<int>(vmaxloc.get()));
  int my_vmaxidx = static_cast<int>(vmaxloc.getLoc()));

The results of these operations will yield the following values:

 * my_vsum == 1000
 * my_vmax == 100
 * my_vminloc == -10
 * my_vminidx == 100 or 500 (depending on order of finalization in parallel)
 * my_vmaxloc == 10
 * my_vmaxidx == 5 or 995 (depending on order of finalization in parallel)

------------------
Reduction Policies
------------------

.. note:: * All RAJA reduction policies are in the namespace ``RAJA``.
          * To guarantee correctness, a reduction policy must be consistent
            with loop execution policy used. That is, a CUDA reduction policy
            must be used when the execution policy is a CUDA policy, an OpenMP
            reduction policy must be used when the execution policy is an
            OpenMP policy, and so on.

* ``seq_reduce``  - Sequential policy for reductions used with sequential and 'loop' execution policies. Currently, RAJA reductions with SIMD execution policies are not defined.

* ``omp_reduce``  - Thread-safe OpenMP reduction policy for use with OpenMP execution policies.

* ``omp_reduce_ordered``  - Thread-safe OpenMP reduction policy that generates reproducible results; e.g., with sum or min/max-loc reductions.

* ``omp_target_reduce``  - Thread-safe OpenMP reduction policy for target offload execution policies (e.g., when using OpenMP4.5 to run on a GPU).

* ``tbb_reduce``  - Thread-safe TBB reduction for use with TBB execution policies.

* ``cuda_reduce`` - Thread-safe reduction policy for use with CUDA execution policies.

* ``cuda_reduce_async`` - Reduction policy for use with CUDA execution policies that may not use explicit cuda synchronization when retrieving its final value.

* ``cuda_reduce_atomic`` - Reduction policy for use with CUDA execution policies that may use CUDA atomic operations in the reduction.

* ``cuda_reduce_atomic_async`` - Reduction policy for use with CUDA execution policies that may not use explicit cuda synchronization when retrieving its final value and which may use CUDA atomic operations in the reduction.

More details on using RAJA reductions can be found in the 
:ref:`dotproduct-label` and :ref:`reductions-label` tutorial sections. 
