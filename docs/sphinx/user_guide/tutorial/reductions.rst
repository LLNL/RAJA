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

.. _reductions-label:

---------------------------------
Reductions
---------------------------------

Key RAJA features shown in this example:

  * ``RAJA::forall`` loop iteration template 
  * ``RAJA::RangeSegment`` iteration space construct
  * RAJA reduction types
  * RAJA reduction policies

The example introduces all of the RAJA supported reduction types: min, max,
sum, min-loc, max-loc, and how they are used. 

The file ``RAJA/examples/ex7-reductions.cpp`` contains the complete 
working example code.
