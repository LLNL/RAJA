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

.. _pi-label:

--------------------------------------------------
Computing Pi with a Reduction or Atomic Operation
--------------------------------------------------

Key RAJA features shown in this example:

  * ``RAJA::forall`` loop iteration template 
  * ``RAJA::RangeSegment`` iteration space construct
  * RAJA reduction types
  * RAJA reduction policies

The example computes an approximation to pi. It illustrates usage of RAJA 
portable reduction types and atomic operations and shows that they are
used similarly for different progamming model backends.

The file ``RAJA/examples/ex8-pi-reduce_vs_atomic.cpp`` contains the complete 
working example code.
