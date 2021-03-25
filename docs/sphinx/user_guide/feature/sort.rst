.. ##
.. ## Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
.. ## and other RAJA project contributors. See the RAJA/COPYRIGHT file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _sort-label:

================
Sorts
================

RAJA provides portable parallel sort operations, which are basic
parallel algorithm building blocks. They are described in this section.

A few important notes:

.. note:: * All RAJA sort operations are in the namespace ``RAJA``.
          * Each RAJA sort operation is a template on an *execution policy*
            parameter. The same policy types used for ``RAJA::forall`` methods
            may be used for RAJA sorts.
          * RAJA sort operations accept an optional *comparator* argument so
            users can perform different types of sort operations. If
            no operator is given, the default is a *less than* operation and
            the result is **non-decreasing**.

Also:

.. note:: * For sorts using the CUDA back-end, RAJA uses the implementations
            provided by the NVIDIA CUB library. For information please see
            :ref:`build-external-tpl <build-external-tpl-label>`.
          * For sorts using the HIP back-end, RAJA uses the implementations
            provided by the AMD rocPRIM library. For information please see
            :ref:`build-external-tpl <build-external-tpl-label>`.
          * The RAJA CUDA and HIP back-ends only support sorting
            arithmetic types using RAJA operators 'less than' and
            'greater than'.

Please see the :ref:`sort-label` tutorial section for usage examples of RAJA
sort operations.

-----------------
Sort Operations
-----------------

In general, a sort operation takes a sequence of numbers ``x`` and a binary
comparison operator ``op`` that forms a strict weak ordering of elements in 
input sequence ``x`` and produces a sequence of numbers ``y`` as output. The 
output sequence is a permutation of the input sequence where each pair of 
elements ``a`` and ``b``, where ``a`` is before ``b`` in the output sequence, 
satisfies ``!(b op a)``. Sorts are stable if they always preserve the order of 
equivalent elements, where equivalent elements satisfy ``!(a op b) && !(b op a)``.

A **stable sort** takes an input sequence ``x`` where a\ :sub:`i` appears 
before a\ :sub:`j` if i < j when a\ :sub:`i` and a\ :sub:`j` are equivalent for 
any i != j. 

   x = { a\ :sub:`0`\, b\ :sub:`0`\, a\ :sub:`1`\, ... }

and calculates the stably sorted output sequence ``y`` that preserves the 
order of equivalent elements. That is, the sorted sequence where element 
a\ :sub:`i` appears before the equivalent element a\ :sub:`j` if i < j:

   y = { a\ :sub:`0`\, a\ :sub:`1`\, b\ :sub:`0`\, ... }

An **unstable sort** may not preserve the order of equivalent elements and
may produce either of the following output sequences:

   y = { a\ :sub:`0`\, a\ :sub:`1`\, b\ :sub:`0`\, ... }

   or

   y = { a\ :sub:`1`\, a\ :sub:`0`\, b\ :sub:`0`\, ... }

---------------------
RAJA Unstable Sorts
---------------------

RAJA unstable sort operations look like the following:

 * ``RAJA::sort< exec_policy >(container)``
 * ``RAJA::sort< exec_policy >(container, comparator)``
 * ``RAJA::sort< exec_policy >(iter, iter + N)``
 * ``RAJA::sort< exec_policy >(iter, iter + N, comparator)``

For example, sorting an array with this sequence of values::

   6 7 2 1 0 9 4 8 5 3 4 9 6 3 7 0 1 8 2 5

with a sequential unstable sort operation:

.. literalinclude:: ../../../../examples/tut_sort.cpp
   :start-after: _sort_seq_start
   :end-before: _sort_seq_end
   :language: C++

produces the ``out`` array with this sequence of values::

   0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9

Note that the syntax is essentially the same as :ref:`scan-label`.
Here, ``container`` is a range of elements and ``iter`` is a random access
iterator to a range of elements. ``container`` and ``iter`` provide access to 
the input sequence and contain the output sequence at the end of sort. The first
and third sort operations listed above will be *non-decreasing* sorts since 
there is no comparator argument given; i.e., the sequences will be reordered 
*in-place* using operator::less. The second and fourth sorts will apply the 
comparator that is passed into the function.

RAJA also provides sort operations that operate on key-value pairs stored
separately:

 * ``RAJA::sort_pairs< exec_policy >(keys_container, vals_container)``
 * ``RAJA::sort_pairs< exec_policy >(keys_container, vals_container, comparator)``
 * ``RAJA::sort_pairs< exec_policy >(keys_iter, keys_iter + N, vals_iter)``
 * ``RAJA::sort_pairs< exec_policy >(keys_iter, keys_iter + N, vals_iter, comparator)``

``RAJA::sort_pairs`` methods generate the same output sequence of keys in 
``keys_container`` or ``keys_iter`` as ``RAJA::sort`` does in ``container`` 
or ``iter`` and reorders the sequence of values in ``vals_container`` or 
``vals_iter`` by permuting the sequence of values in the same manner as the 
sequence of keys; i.e. the sequence of pairs is sorted based on comparing 
their keys. 

.. note:: The comparator used in ``RAJA::sort_pairs`` only compares keys.

---------------------
RAJA Stable Sorts
---------------------

RAJA stable sorts are essentially the same as unstable sorts:

 * ``RAJA::stable_sort< exec_policy >(container)``
 * ``RAJA::stable_sort< exec_policy >(container, comparator)``
 * ``RAJA::stable_sort< exec_policy >(iter, iter + N)``
 * ``RAJA::stable_sort< exec_policy >(iter, iter + N, comparator)``

RAJA also provides stable sort pairs that operate on key-value pairs stored
separately:

 * ``RAJA::stable_sort_pairs< exec_policy >(keys_container, vals_container)``
 * ``RAJA::stable_sort_pairs< exec_policy >(keys_container, vals_container, comparator)``
 * ``RAJA::stable_sort_pairs< exec_policy >(keys_iter, keys_iter + N, vals_iter)``
 * ``RAJA::stable_sort_pairs< exec_policy >(keys_iter, keys_iter + N, vals_iter, comparator)``

.. _sortops-label:

--------------------
RAJA Comparison Operators
--------------------

RAJA provides two operators that can be used to produce different ordered sorts:

  * ``RAJA::operators::less<T>``
  * ``RAJA::operators::greater<T>``

.. note:: All RAJA comparison operators are in the namespace ``RAJA::operators``.

-------------------
Sort Policies
-------------------

For information about RAJA execution policies to use with sort operations,
please see :ref:`policies-label`.


