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

================
Scans
================

RAJA provides portable parallel scan operations, which are basic 
parallel algorithm building blocks. They are described in this section.

-----------------
Scan Operations
-----------------

All scan operations take a sequence of numbers 'x' and a binary 
associative operator 'op' as input and produce another sequence of 
numbers 'y' as output. Each element of the output sequence is formed by 
applying the operator to a subset of the input. Scans come in 
two flavors: *inclusive* and *exclusive*.

An **inclusive scan** takes the input sequence

   x = { x\ :sub:`0`\, x\ :sub:`1`\, x\ :sub:`2`\, ... }

and calculates the output sequence::

   y = { y0, y1, y2, ... }

using the recursive definition::

  y0 = x0
  yi = y(i-1) op xi for each i > 0

An **exclusive scan** is similar, but the output of an exclusive scan is 
differend from the output of an inclusive scan in two ways. First, the first 
element of the output is the identity of the operator used. Second, the 
rest of the output sequence is the same as inclusive scan, but shifted one 
position to the right; i.e.,::

  y0 = identity(op)
  yi = y(i-1) op x(i-1) for each i > 0

If you would like more information about scan operations, a good overview of 
what they are and why they are useful can be found here 
`Blelloch Scan Lecture Notes <https://www.cs.cmu.edu/~blelloch/papers/Ble93.pdf>`_. A nice presentation that describes how parallel scans are implemented is `Va Tech Scan Lecture <http://people.cs.vt.edu/yongcao/teaching/cs5234/spring2013/slides/Lecture10.pdf>`_

A few important notes:

.. note:: * All RAJA scan operations are in the namespace ``RAJA``.
          * Each RAJA scan operation is templated on an *execution policy*.
            The same policy types used for ``RAJA::forall`` methods are used
            for RAJA scans. 
          * RAJA scan operations accept an optional *operator* argument so 
            users can perform different types of scan-type operations. If
            no operator is given, the default is a 'plus' operation and
            the reesult is a **prefix-sum**.

Also:

.. note:: For scans using the CUDA back-end, RAJA uses the implementations
          provided by the NVIDIA Thrust library. For better performance, one
          can enable the NVIDIA cub library for scans by setting the CMake
          variable ``CUB_DIR`` to the location of the cub library on your
          system when CUDA is enabled.

RAJA supports the following scan operations:

----------------
Inclusive Scans
----------------

 * ``RAJA::inclusive_scan< exec_policy >(in, in + N, out)``
 * ``RAJA::inclusive_scan< exec_policy >(in, in + N, out, operator)``

Here, 'in' and 'out' are pointers to arrays of some numeric scalar type. The
scalar type for both arrays must be the same for each scan operation. The
'in' array is the input data to the scan and 'out' is the result of the scan
operation. The first scan operation above will be a prefix-sum since there 
is no operator argument given; i.e., the output array will contain partial 
sums of the input array. The second scan will apply the operator that is passed.

 * ``RAJA::inclusive_scan_inplace< exec_policy >(in, in + N)``
 * ``RAJA::inclusive_scan_inplace< exec_policy >(in, in + N, <operator>)``

In-place scans operate differ from the previous scans in that they do not
take separate input and output arrays. They generate the result of the scan
operation *in-place* in the input array.

----------------
Exclusive Scans
----------------




Please see the :ref:`scan-label` tutorial section for usage examples of RAJA
scan operations.
