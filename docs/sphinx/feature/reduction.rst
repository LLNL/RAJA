.. _reduction::

==========
Reductions
==========

Computational patterns such as accumulating values, or finding the maximum/minimum
of an array falls under a type of computation known as reductions. In threading paradigms
this requires multiple threads writting out to the same scalar value. RAJA enables thread
safe operations with the introduction of ``RAJA::Reduce`` variables

----------------
Reduce Variables
----------------

* ReduceSum : Provides a thread safe variable to accumulate a sum. Basic usage ``RAJA::ReduceSum<reduce_policy, data_type> RAJA_resI2(0.0)``

* ReduceMin : Provides a thread safe variable to compute the minimum. Basic usage ``RAJA::ReduceSum<reduce_policy, data_type> RAJA_resI2(0.0)``

* ReduceMax : Provides a thread safe variable to compute the maximum. Basic usage ``RAJA::ReduceSum<reduce_policy, data_type> RAJA_resI2(0.0)``

* ReduceMinLoc : Returns the index of the maximum 

* ReduceMaxLoc : Returns the index of the minimum

------------------
Reduce Policies
------------------

* seq_reduce  : Creates a thread safe variable under Sequential or Loop policies

* omp_reduce  : Creates a thread safe variable under OpenMP policies

* tbb_reduce  : Creates a thread safe variable under TBB Policies 

* cuda_reduce : Creates a thread safe variable under CUDA policies

