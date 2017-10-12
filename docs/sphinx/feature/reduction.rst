.. _reduction::

==========
Reductions
==========

Computational patterns such as accumulating values, or finding the maximum/minimum
of an array falls under a type of computation known as reductions. In threading paradigms
this requires multiple threads writting out to the same scalar value. RAJA enables thread
safe operations with the introduction of ``RAJA::Reduce`` variables

Reduce Variables:

* ReduceSum : 
* ReduceMin : 
* ReduceMax : 
* ReduceMinLoc (CUDA specific) : 
* ReduceMaxLoc (CUDA specific) : 

Reduction Policies:

* seq_reduce
* omp_reduce
* tbb_reduce  
* cuda_reduce

