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

.. _addvectors-label:

----------------
Vector Addition
----------------

Main RAJA features discussed:

  * ``RAJA::forall`` loop traversal template
  * ``RAJA::RangeSegment`` iteration space construct
  * RAJA execution policies


In this example, we add two vectors 'A' and 'B' of length N and
store the result in vector 'C'. We assume that we have allocated and 
initialized arrays holding the data for the vectors. Then, a typical 
C-style loop to add the vectors is:

.. literalinclude:: ../../../examples/example-add-vectors.cpp
                    :lines: 94-96

^^^^^^^^^^^^^^^^^^^^^
Converting to RAJA
^^^^^^^^^^^^^^^^^^^^^

The RAJA version of this loop requires specifying an execution policy
(for more information, see :ref:`policies-label`) and an iteration space. Here,
we use a ``RAJA::RangeSegment``, which describes a contiguous sequence of 
integral values [0, N) (for more information, see :ref:`index-label`). 

.. literalinclude:: ../../../examples/example-add-vectors.cpp
                    :lines: 107-112

^^^^^^^^^^^^^^^^^^^^^
Running on a GPU
^^^^^^^^^^^^^^^^^^^^^

By using a different execution policy, we can also run this loop on a CUDA
GPU. Note that the user is responsible for making sure that the data arrays
are properly allocated and initialized on the device. This can be done using
explicit device allocation and copying from host memory or via CUDA unified
memory, if available (for more information, see :ref:`plugins-label`). 
Since the lambda function defining the loop body will be passed to a device
kernel, the lambda must be decorated with the ``__device__`` attribute when
the lambda is defined.

.. literalinclude:: ../../../examples/example-add-vectors.cpp
                    :lines: 138-143

RAJA CUDA execution policies are templated on the number of threads per block,
giving users control of how device threads are managed. the number of threads
per block is optional since the policies provide a reasonable default for most
cases. However, this may need to be changed for performance reasons or when 
there is a large number of variables in the loop body requiring many device
registers.

The file ``RAJA/examples/example-add-vectors.cpp`` contains the complete 
working example code.
