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

.. _scan-label:

====
Scan
====

The scan operation (or prefix sum) is an operation such that provided a one-dimensional array :math:`x`,
the resulting array :math:`y` is given as

.. math::

   y[i] \leftarrow \sum^{i}_{j=0} x[j].

The example above illustrates the so-called inclusive sum because each element i of
the result :math:`y_i` is the sum of all elements including :math:`i` in :math:`x`. An exclusive sum
would accumulate values up to but not including :math:`i`. RAJA provides a simple interface for carrying out these prefix operations. 


The basic usage is as follows, intializing arrays :math:`x`, :math:`y`

.. literalinclude:: ../../../examples/example-scan.cpp
   :lines: 26-31

a RAJA inclusive prefix sum can be carried out through the following iterface

.. literalinclude:: ../../../examples/example-scan.cpp
   :lines: 36

where ``execute_policy`` is one of the RAJA policies. Similary, an exclusive prefix sum may
be carried out through the following interface

.. literalinclude:: ../../../examples/example-scan.cpp
   :lines: 43


In each of the interfaces the first argument corresponds to a pointer to the start of the array, 
a pointer to the end of the array, and a pointer to the output. 

A full working version is maybe be found in the example folder ``example-scan.cpp``






