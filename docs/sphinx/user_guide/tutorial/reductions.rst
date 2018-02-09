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

---------------------------------
Reductions
---------------------------------

Key RAJA features shown in this example:

  * ``RAJA::forall`` loop iteration template 
  * ``RAJA::RangeSegment`` iteration space construct
  * RAJA reduction types
  * RAJA reduction policies

In the :ref:`dotproduct-label` example, we showed how to use the RAJA sum reduction type.
The example introduces all of the RAJA supported reduction types: min, max,
sum, min-loc, max-loc, and shows how they are used. All RAJA reduction types
are defined and used similarly.

.. note:: Multiple RAJA reductions can be combined in any RAJA kernel execution
          constructs, and the reductions can be combined with any other
          kernel operations. 

We start by allocating an array, using CUDA Unified Memory if CUDA is enabled,
and initializing its values in a manner that makes it easy to show what the
different reduction types do:

.. literalinclude:: ../../../../examples/ex7-reductions.cpp
                    :lines: 51-76

We also define a range segment that defines the iteration space over the array:

.. literalinclude:: ../../../../examples/ex7-reductions.cpp
                    :lines: 93-93

With these parameters and data initialization, all the code examples 
presented below will generate the following results:

 * the sum will be zero
 * the min will be -100
 * the max will be 100
 * the min loc will be N/2
 * the max loc will be N/2 + 1

.. note:: Each RAJA reduction type requires a reduction policy that must 
          be compatible with the execution policy for the kernel in which 
          it is used.

A sequential kernel that exercises all RAJA sequential reduction types is:
 
.. literalinclude:: ../../../../examples/ex7-reductions.cpp
                    :lines: 99-118

Note that each reduction takes an initial value at construction and that 
reduced values are retrieved after the kernel completes  by using an 
appropriate 'get' method. The 'get()' method returns the reduced value for
each reduction type; the 'getLoc()' method returns the index at which the 
reduced value was observed for min-loc and max-loc reductions

For parallel multi-threading execution via OpenMP, the example can be run 
by replacing the execution and reduction policies policies with:

.. literalinclude:: ../../../../examples/ex7-reductions.cpp
                    :lines: 134-135

Similarly, the kernel containing the reductions can be run in parallel
on a CUDA GPU using these policies:

.. literalinclude:: ../../../../examples/ex7-reductions.cpp
                    :lines: 170-171

Note that for CUDA reductions to execute correctly, the thread block size 
in the reduction policy must match that which is used in the CUDA 
execution policy.

The file ``RAJA/examples/ex7-reductions.cpp`` contains the complete 
working example code.
