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
Reduction Operations
====================

RAJA does not provide separate loop kernel execution methods for
reduction operations like some other C++ template-based programming models do.
Instead RAJA provides reduction types that allow users to perform reduction 
operations with ``RAJA::forall`` and ``RAJA::kernel`` methods in a portable, 
thread-safe manner. Users can also use as many reductions in a loop kernel
as they need. Available RAJA reduction types are described in this section.

.. note:: * All RAJA reduction types are in the namespace ``RAJA``.
          * Each RAJA reduction type is templated on an *reduction policy* 
            and a value type for the reduction variable. The reduction policy
            **must be compatible** with the loop kernel execution policy
            where the reduction is used.
          * Each RAJA reduction type accepts an initial reduction value at
            construction.
          * Each RAJA reduction type has 'get' methods to access its reduced
            value after kernel execution completes.

Other examples of RAJA reductions can be found in the 
:ref:`reductions-label` tutorial sections. 

----------------
Reduction Types
----------------

* ``ReduceSum< reduce_policy, data_type >`` - Sum of values.

* ``ReduceMin< reduce_policy, data_type >`` - Min value.

* ``ReduceMax< reduce_policy, data_type >`` - Max value.

* ``ReduceMinLoc< reduce_policy, data_type >`` - Min value and a loop index where the minimum was found.

* ``ReduceMaxLoc< reduce_policy, data_type >`` - Max value and a loop index where the maximum was found.

.. note:: * When ``ReduceMinLoc`` and ``ReduceMaxLoc`` are used in a sequential
            execution context, the loop index where the min/max was found is
            the first index where the min/max occurs.
          * When these reductions are used in a parallel execution context, 
            the loop index given for the min/max may be any index where the
            min/max occurs. 

Here's a simple RAJA reduction example that uses a sum reduction type and a 
min-loc reduction type::

  const int N = 1000;

  //
  // Initialize array of length N with all ones. Then, set some other
  // values to make the example mildly interesting...
  //
  int vec[N] = {1};
  vec[100] = -10; vec[500] = -10;

  // Create sum and min-loc reduction object with initial values
  RAJA::ReduceSum< RAJA::omp_reduce, int > vsum(0);
  RAJA::ReduceMinLoc< RAJA::omp_reduce, int > vminloc(100, -1);

  RAJA::forall<RAJA::omp_parallel_for_exec>( RAJA::RangeSegment(0, N),
    [=](RAJA::Index_type i) {

    vsum += vec[i] ;
    vminloc.minloc( vec[i] ) ;

  });

  int my_vsum = static_cast<int>(vsum.get());

  int my_vminloc = static_cast<int>(vminloc.get()));
  int my_vminidx = static_cast<int>(vminloc.getLoc()));

The results of these operations will yield the following values:

 * my_vsum == 1000
 * my_vminloc == -10
 * my_vminidx == 100 or 500 (depending on order of finalization in parallel)

------------------
Reduction Policies
------------------

.. note:: * All RAJA reduction policies are in the namespace ``RAJA``.
          * To guarantee correctness, a reduction policy must be consistent
            with loop execution policy used. For example, a CUDA reduction 
            policy must be used when the execution policy is a CUDA policy, 
            an OpenMP reduction policy must be used when the execution policy 
            is an OpenMP policy, and so on.
          * **RAJA reductions used with SIMD execution policies are not 
            guranteed to generate correct results.**

* ``seq_reduce``  - Sequential policy for reductions used with sequential and 'loop' execution policies. 

* ``omp_reduce``  - Thread-safe OpenMP reduction policy for use with OpenMP execution policies.

* ``omp_reduce_ordered``  - Thread-safe OpenMP reduction policy that generates reproducible results; e.g., with sum or min-loc/max-loc reductions.

* ``omp_target_reduce``  - Thread-safe OpenMP reduction policy for target offload execution policies (i.e., when using OpenMP4.5 to run on a GPU).

* ``tbb_reduce``  - Thread-safe TBB reduction for use with TBB execution policies.

* ``cuda_reduce`` - Thread-safe reduction policy for use with CUDA execution policies.

* ``cuda_reduce_async`` - Reduction policy for use with CUDA execution policies that may not use cuda device synchronization when retrieving final reduction value.

* ``cuda_reduce_atomic`` - Reduction policy for use with CUDA execution policies that may use CUDA atomic operations in the reduction.

* ``cuda_reduce_atomic_async`` - Reduction policy for use with CUDA execution policies that may not use cuda device synchronization when retrieving final reduction value and which may use CUDA atomic operations in the reduction.
