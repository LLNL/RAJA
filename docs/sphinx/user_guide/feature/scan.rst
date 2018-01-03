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
the resulting array :math:`y` is given as:

.. math::

   y[i] \leftarrow \sum^{i}_{j=0} x[j].

The example above illustrates the so-called inclusive sum because each :math:`i` of
the result :math:`y_i` is computed via the sum of all elements including
:math:`i` in :math:`x`. An exclusive sum accumulates values up to but not
including :math:`i`. 

--------------
Scan Interface
--------------

As the scan operation comes in two variants (inclusive/exclusive) RAJA provides an interface for both. 

.. note:: * All RAJA scan operations are in the namespace ``RAJA``.

The inclusive scan is invoked by the following interface

* ``RAJA::exclusive_scan<execute_policy>(xptr, xptr+N, yptr)``

similarly the exclusive scan is invoked via the following interface

* ``RAJA::inclusive_scan<execute_policy>(xptr, xptr+N, yptr)``

Here ``xptr``, and ``yptr`` corresponds to pointers to the input, and output arrays which both have length ``N``. 

.. literalinclude:: ../../../../examples/example-scan.cpp
                    :lines: 25-55

The full working example is found in ``example-scan.cpp``.