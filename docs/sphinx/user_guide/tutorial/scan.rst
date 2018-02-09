.. ##
.. ## Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
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

--------------------------------------------------
Parallel Scan Operations
--------------------------------------------------

Key RAJA features shown in this example:

  * ``RAJA::inclusive_scan`` operation
  * ``RAJA::inclusive_scan_inplace`` operation
  * ``RAJA::exclusive_scan`` operation
  * ``RAJA::exclusive_scan_inplace`` operation
  * RAJA operators for different types of scans; e.g., plus, minimum, maximum, etc.

In the example, we demonstrate a variety of scan operations supported by RAJA.
We show how different scan operations can be performed by passing different
RAJA operators to the RAJA scan template methods. Each operator is a template
type, where the template argument is the type of the values it operates on.

This discussion assumes the reader is familiar with parallel scan operations
and how they are applied. If you are unfamiliar with scan operations or need
a refresher, a good explanation of what scan operations are why they are
useful can be found here `Blelloch Scan Lecture Notes <https://www.cs.cmu.edu/~blelloch/papers/Ble93.pdf>`. A nice presentation that describes how scans are
parallelized is `Va Tech Scan Lecture <http://people.cs.vt.edu/yongcao/teaching/cs5234/spring2013/slides/Lecture10.pdf>`_ 

.. note:: For scans using the CUDA back-end, RAJA uses the implementations
          provided by the NVIDIA Thrust library. For better performance, one
          can enable the NVIDIA cub library for scans by setting the CMake
          variable ``CUB_DIR`` to the location of the cub library on your
          system when CUDA is enabled.

In the following discussion, we present examples of RAJA sequential, OpenMP,
and CUDA scan operations. All examples use the same integer arrays for input
and output values. We set the input array as follows:

.. literalinclude:: ../../../../examples/ex9-scan.cpp
                    :lines: 67-83

This generates the following sequence of values in the 'in' array::

   3 -1 2 15 7 5 17 9 6 18 1 10 0 14 13 4 11 12 8 16

.. note:: If no operator is passed to a RAJA scan method, the operator
          ``RAJA::operators::plus`` is used by default. The result is a 
          `prefix-sum`.

^^^^^^^^^^^^^^^^
Inclusive Scans
^^^^^^^^^^^^^^^^

A sequential inclusive scan operation is performed by:

.. literalinclude:: ../../../../examples/ex9-scan.cpp
                    :lines: 93-93

Note that entries of the 'out' array are set to a prefix-sum based on the 'in'
array since no operator is passed to the scan method. The resulting 'out' 
array contains the values::

   3 2 4 19 26 31 48 57 63 81 82 92 92 106 119 123 134 146 154 170

We can be explicit about the operation used in the scan by passing the 
'plus' operator to the scan method:

.. literalinclude:: ../../../../examples/ex9-scan.cpp
                    :lines: 103-104

The result in the 'out' array is the same.

An inclusive parallel scan operation using OpenMP multi-threading is
accomplished like this (we are explicit with the operation the scan will use):

.. literalinclude:: ../../../../examples/ex9-scan.cpp
                    :lines: 156-157

As is commonly done with RAJA, the only difference between this code and
the previous one is that the execution policy is different. If we want to 
run the scan on a GPU using CUDA, we would use a CUDA execution policy. This
will be shown shortly.

^^^^^^^^^^^^^^^^
Exclusive Scans
^^^^^^^^^^^^^^^^

A sequential exclusive scan (plus) operation is performed by:

.. literalinclude:: ../../../../examples/ex9-scan.cpp
                    :lines: 114-115

This generates the following sequence of values in the output array::

   0 3 2 4 19 26 31 48 57 63 81 82 92 92 106 119 123 134 146 154

Note that the exclusive scan result is different than the inclusive scan 
result in two ways. The first entry in the result is the `identity` of the
operator used (here, it is zero, since the operator is 'plus') and, after
that, the output sequence is shifted one position to the right.

Running the same scan operation on a GPU using CUDA is done by:

.. literalinclude:: ../../../../examples/ex9-scan.cpp
                    :lines: 199-200

Note that we pass the number of threads per CUDA thread block as the template
argument to the CUDA execution policy.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In-place Scans and Other Operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In-place scan operations are identical to the ones we have shown. However, the
result is generated in the input array directly so only one array is passed
to in-place scan methods.

Here is a sequential inclusive in-place scan that uses the 'minimum' operator:

.. literalinclude:: ../../../../examples/ex9-scan.cpp
                    :lines: 125-128

Note that, before the scan, we copy the input array into the output array so 
the result is generated in the output array. Doing this, we avoid having to
re-initialize the input array to use it in other examples. 

This generates the following sequence in the output array::

   3 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1

Here is a sequential exclusive in-place scan that uses the 'maximum' operator:

.. literalinclude:: ../../../../examples/ex9-scan.cpp
                    :lines: 138-141

This generates the following sequence in the output array::

   -2147483648 3 3 3 15 15 15 17 17 17 18 18 18 18 18 18 18 18 18 18

Note that the first value in the result is the negative of the max int value;
i.e., the identity of the maximum operator.

As you may expect at this point, running an exclusive in-place prefix-sum
operation using OpenMP is accomplished by: 

.. literalinclude:: ../../../../examples/ex9-scan.cpp
                    :lines: 167-170

This generates the following sequence in the output array (as expected)::

   0 3 2 4 19 26 31 48 57 63 81 82 92 92 106 119 123 134 146 15

Lastly, we show a parallel inclusive in-place prefix-sum operation using CUDA:

.. literalinclude:: ../../../../examples/ex9-scan.cpp
                    :lines: 187-190

The file ``RAJA/examples/ex9-scan.cpp`` contains the complete 
working example code.
