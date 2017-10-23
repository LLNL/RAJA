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

.. _scan::

====
Scan
====

RAJA also provides built in functions to carryout a prefix sum. A prefix sum or scan is an operation
such that provided a one-dimensional array x, the resulting array y is given as

.. math::

   y[i] \leftarrow \sum^{i}_{j=0} x[j]

The example above illustrates the so-called exclusive sum because each element i of
the result is the sum of all elements up to but not including i. An inclusive sum all elements up 
to i are summed. 

RAJA provides a simple interface for carrying out these prefix operations. Intializing arrays x,y 

.. literalinclude:: ../../../examples/example-scan.cpp
   :lines: 26-32

a RAJA inclusive prefix sum can be carried out through the following iterface

.. literalinclude:: ../../../examples/example-scan.cpp
   :lines: 37

where ``execute_policy`` is one of the RAJA policies. Similary an exclusive prefix sum may
be carried out through the following interface

.. literalinclude:: ../../../examples/example-scan.cpp
   :lines: 44


In each of the interfaces the first argument corresponds to a pointer to the start of the array, 
a pointer to the end of the array, and a pointer to the output. 

A full working version is maybe be found in the example folder ``example-scan.cpp``








