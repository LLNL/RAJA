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

.. _nestedreorder-label:

---------------------------------
Nested Loops and Loop Interchange
---------------------------------

Key RAJA features shown in this example:

  * ``RAJA::RangeSegment`` iteration space construct
  * ``RAJA::nested::forall`` loop iteration templates 
  * RAJA nested loop execution policies
  * Nested loop re-ordering (i.e., loop interchange)
  * RAJA strongly-types indices.

This example introduces RAJA nested loop constructs and shows how to 
reorder the levels in a loop nest by re-ordering nested policy arguments.
The example does not perform any real computation; each kernel simply
prints out the loop indices in the order that the iteration spaces are
traversed. Thus, only sequential execution policies are used.

The file ``RAJA/examples/ex5-nested-loop-reorder.cpp`` contains the complete 
working example code.
