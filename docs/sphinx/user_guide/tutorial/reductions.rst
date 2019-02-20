.. ##
.. ## Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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

  * ``RAJA::forall`` loop execution template 
  * ``RAJA::RangeSegment`` iteration space construct
  * RAJA reduction types
  * RAJA reduction policies

In the :ref:`dotproduct-label` example, we showed how to use the RAJA sum 
reduction type. The following example uses all supported RAJA reduction types: 
min, max, sum, min-loc, max-loc.

.. note:: Multiple RAJA reductions can be combined in any RAJA loop kernel 
          execution method, and reduction operations can be combined with 
          any other kernel operations. 

We start by allocating an array (the memory manager in the example uses 
CUDA Unified Memory if CUDA is enabled) and initializing its values in a 
manner that makes the example mildly interesting and able to show what the 
different reduction types do. Specifically, the array is initialized to
a sequence of alternating values ('1' and '-1'). Then, two values near
the middle of the array are set to '-100' and '100':

.. literalinclude:: ../../../../examples/tut_reductions.cpp
                    :lines: 51-76

We also define a range segment to iterate over the array:

.. literalinclude:: ../../../../examples/tut_reductions.cpp
                    :lines: 93-93

With these parameters and data initialization, all the code examples 
presented below will generate the following results:

 * the sum will be zero
 * the min will be -100
 * the max will be 100
 * the min loc will be N/2
 * the max loc will be N/2 + 1

A sequential kernel that exercises all RAJA sequential reduction types is:
 
.. literalinclude:: ../../../../examples/tut_reductions.cpp
                    :lines: 99-126

Note that each reduction object takes an initial value at construction. Also,
within the kernel, updating each reduction is done via an operator or method
that is basically what you would expect (i.e., '+=' for sum, 'min()' for min,
etc.). After the kernel executes, the reduced value computed by each reduction 
object is retrieved after the kernel by calling a 'get()' method on the 
reduction object. The min-loc/max-loc index values are obtained using 
'getLoc()' methods.

For parallel multi-threading execution via OpenMP, the example can be run 
by replacing the execution and reduction policies with:

.. literalinclude:: ../../../../examples/tut_reductions.cpp
                    :lines: 134-135

Similarly, the kernel containing the reductions can be run in parallel
on a CUDA GPU using these policies:

.. literalinclude:: ../../../../examples/tut_reductions.cpp
                    :lines: 170-171

.. note:: Each RAJA reduction type requires a reduction policy that must 
          be compatible with the execution policy for the kernel in which 
          it is used.

The file ``RAJA/examples/tut_reductions.cpp`` contains the complete 
working example code.
