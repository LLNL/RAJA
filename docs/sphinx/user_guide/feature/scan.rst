.. ##
.. ## Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
.. ## and other RAJA project contributors. See the RAJA/COPYRIGHT file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _scan-label:

================
Scans
================

RAJA provides portable parallel scan operations, which are basic 
parallel algorithm building blocks. They are described in this section.

A few important notes:

.. note:: * All RAJA scan operations are in the namespace ``RAJA``.
          * Each RAJA scan operation is a template on an *execution policy*
            parameter. The same policy types used for ``RAJA::forall`` methods 
            may be used for RAJA scans. 
          * RAJA scan operations accept an optional *operator* argument so 
            users can perform different types of scan operations. If
            no operator is given, the default is a 'plus' operation and
            the result is a **prefix-sum**.

Also:

.. note:: For scans using the CUDA back-end, RAJA uses the NVIDIA cub library
          internally, which is available in the RAJA source repository as a 
          Git submodule. The CMake variable ``CUB_DIR`` will be automatically 
          set to the location of the cub library when CUDA is enabled. Details
          for using a different version of the cub library are available in
          the :ref:`getting_started-label` section.

Please see the :ref:`scan-label` tutorial section for usage examples of RAJA
scan operations.

-----------------
Scan Operations
-----------------

In general, a scan operation takes a sequence of numbers 'x' and a binary 
associative operator 'op' as input and produces another sequence of 
numbers 'y' as output. Each element of the output sequence is formed by 
applying the operator to a subset of the input. Scans come in 
two flavors: *inclusive* and *exclusive*.

An **inclusive scan** takes the input sequence

   x = { x\ :sub:`0`\, x\ :sub:`1`\, x\ :sub:`2`\, ... }

and calculates the output sequence:

   y = { y\ :sub:`0`\, y\ :sub:`1`\, y\ :sub:`2`\, ... }

using the recursive definition

   y\ :sub:`0`\ = x\ :sub:`0`

   y\ :sub:`i`\ = y\ :sub:`i-1`\ op x\ :sub:`i`\, for each i > 0

An **exclusive scan** is similar, but the output of an exclusive scan is 
different from the output of an inclusive scan in two ways. First, the first 
element of the output is the identity of the operator used. Second, the 
rest of the output sequence is the same as inclusive scan, but shifted one 
position to the right; i.e.,

   y\ :sub:`0`\ = op\ :sub:`identity`

   y\ :sub:`i`\ = y\ :sub:`i-1` op x\ :sub:`i-1`\, for each i > 0

If you would like more information about scan operations, a good overview of 
what they are and why they are useful can be found in 
`Blelloch Scan Lecture Notes <https://www.cs.cmu.edu/~blelloch/papers/Ble93.pdf>`_. A nice presentation that describes how parallel scans are implemented is `Va Tech Scan Lecture <http://people.cs.vt.edu/yongcao/teaching/cs5234/spring2013/slides/Lecture10.pdf>`_

---------------------
RAJA Inclusive Scans
---------------------

RAJA inclusive scan operations look like the following:

 * ``RAJA::inclusive_scan< exec_policy >(in, in + N, out)``
 * ``RAJA::inclusive_scan< exec_policy >(in, in + N, out, operator)``

Here, 'in' and 'out' are pointers to arrays of some numeric scalar type whose
elements are the input and output sequences of the scan, respectively. The
scalar type must be the same for both arrays. The first scan operation above 
will be a *prefix-sum* since there is no operator argument given; i.e., the 
output array will contain partial sums of the input array. The second scan 
will apply the operator that is passed.

RAJA also provides *in-place* scans:  

 * ``RAJA::inclusive_scan_inplace< exec_policy >(in, in + N)``
 * ``RAJA::inclusive_scan_inplace< exec_policy >(in, in + N, <operator>)``

An in-place scan generates the same output sequence as a non-inplace scan.
However, an in-place scan does not take separate input and output arrays and
the result of the scan operation will appear *in-place* in the input array.

---------------------
RAJA Exclusive Scans
---------------------

Using RAJA exclusive scans is essentially the same as for inclusive scans:

 * ``RAJA::exclusive_scan< exec_policy >(in, in + N, out)``
 * ``RAJA::exclusive_scan< exec_policy >(in, in + N, out, operator)``

and

 * ``RAJA::exclusive_scan_inplace< exec_policy >(in, in + N)``
 * ``RAJA::exclusive_scan_inplace< exec_policy >(in, in + N, <operator>)``

.. _scanops-label:

--------------------
RAJA Scan Operators
--------------------

RAJA provides a variety of operators that can be used to perform different
types of scans, such as:

  * ``RAJA::operators::plus<T>``
  * ``RAJA::operators::minus<T>``
  * ``RAJA::operators::multiplies<T>``
  * ``RAJA::operators::divides<T>``
  * ``RAJA::operators::minimum<T>``
  * ``RAJA::operators::maximum<T>``

.. note:: * All RAJA scan operators are in the namespace ``RAJA::operators``.

-------------------
Scan Policies
-------------------

For information about RAJA execution policies to use with scan operations, 
please see :ref:`policies-label`.


