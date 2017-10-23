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

.. _addVectors::

============
Add Vectors
============

As a first example we consider vector addition. Here two vectors A, and B, of length N are added together
and the result is stored in a third vector, C. The C++ version of this loop takes the following form

.. literalinclude:: ../../../examples/example-add-vectors.cpp
                    :lines: 93-96

---------------------
1. Converting to RAJA
---------------------

The construction of a RAJA analog begins by first specifying an execution policy
(for more info see :ref:`ref-policy`) and constructing an iteration space. For this example we can generate
an iteration space composed of a contiguous sequence of numbers by using ``RAJA::RangeSegment`` (for more info see :ref:`ref-index`). 

.. literalinclude:: ../../../examples/example-add-vectors.cpp
                    :lines: 107-112

-------------------
2. RAJA on the GPU
-------------------

By swapping out execution policies we can target different backends, the caveat being that the developer is reponsible for memory management (for more info see :ref:`ref-plugins`). Furthermore, using the cuda backend requires
the ``__device__`` decorator on the lambda. 

.. literalinclude:: ../../../examples/example-add-vectors.cpp
                    :lines: 138-143

Lastly, invoking the CUDA execution policy requires specifying the number of threads per block.
A full working version ``example-add-vectors.cpp`` may be found in the example folder.
