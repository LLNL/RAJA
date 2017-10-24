.. ##
.. ## Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
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

* ReduceSum : Provides a thread safe variable to accumulate a sum

* ReduceMin : Provides a thread safe variable to compute the minimum

* ReduceMax : Provides a thread safe variable to compute the maximum

* ReduceMinLoc : Returns the index of the maximum 

* ReduceMaxLoc : Returns the index of the minimum

------------------
Reduce Policies
------------------

* seq_reduce  : Creates a thread safe variable under Sequential or Loop policies

* omp_reduce  : Creates a thread safe variable under OpenMP policies

* tbb_reduce  : Creates a thread safe variable under TBB Policies 

* cuda_reduce : Creates a thread safe variable under CUDA policies

Basic usage is illustrated in ``example-reduction.cpp``.