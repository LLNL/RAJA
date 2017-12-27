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

.. _dotproduct-label:

------------------
Vector Dot Product
------------------

Key RAJA features shown in this example:

  * ``RAJA::forall`` loop traversal template
  * ``RAJA::RangeSegment`` iteration space construct
  * RAJA execution policies
  * RAJA reduction types


In the example, we compute the vector dot product, 'dot = (a,b)', where 
'a' and 'b' are two vectors length N and 'dot' is a scalar. A typical
C-style loop for the dot product is: 

.. literalinclude:: ../../../../examples/ex2-dot-product.cpp
                    :lines: 81-85

The file ``RAJA/examples/ex2-dot-product.cpp`` contains the complete 
working example code.
