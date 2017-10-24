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

.. _policies::
.. _ref-policy:

===============
Policy Appendix
===============

--------------------
Serial/SIMD Policies
--------------------

* seq_exec  : Enforces sequential iterations
* loop_exec : Vectorizies if compiler belives it is appropriate
* simd_exec : Introduces compiler hints for vectorizations

---------------
OpenMP Policies
---------------

* omp_for_exec : Instructs the compiler to distribute loop iterations within the threads
* omp_for_nowait_exec : Removes synchronization within threaded regions
* omp_for_static : Each thread receives approximately the same number of iterations
* omp_parallel_exec : Creates a parallel region
* omp_parallel_for_exec : Specifies a parallel region containing one or more associated loops
* omp_parallel_segit : 
* omp_parallel_for_segit : 
* omp_collapse_nowait_exec : 
* omp_reduce : 
* omp_reduce_ordered : 

----------------------
OpenMP Target Policies
----------------------

* omp_target_parallel_for_exec :
* omp_target_reduce :   
  
------------
TBB Policies
------------ 

* tbb_for_exec : 
* tbb_for_static :
* tbb_for_dynamic : 
* tbb_segit : 
* tbb_reduce : 

-------------
CUDA Policies
-------------

* cuda_threadblock_x_exec<int THREADS>
* cuda_threadblock_y_exec<int THREADS>
* cuda_threadblock_z_exec<int THREADS>
* cuda_reduce <int THREADS>


