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

.. _index-label:

====================
Indices and Segments
====================

-------
Indices
-------

Although RAJA may utilize any loop counter, the recommended loop counter is the ``RAJA::Index_type``. The ``RAJA::Index_type`` 
is a 64-bit loop counter which makes it easy for the compiler to carryout optimizations.

-------------
RAJA Segments
-------------

RAJA introduces a family of segments which serve as containers for iterations spaces. 
The fundamental segment types are ``RAJA::RangeSegment``, and ``RAJA::TypedRangeSegment``; the former is constructed 
via :: 

    using RAJA::RangeSegment = RAJA::TypedRangeSegment<RAJA::Index_type>

Under the RAJA programming model, the purpose of these containers is to generate a contiguous sequence of numbers and more fundamentally,
to serve as the 

   RAJA::forall<exec_policy>(TypedRangeSegment<T>(begin, end), [=] (RAJA::Index_type i)) {
     //loop boody
   }

.. note:: * All segment objects are found in the namespace RAJA
          * TypedRangeStrideSegment::iterator is a random access iterator
          * TypedRangeStrideSegment allows for positive or negative strides, but a stride of zero is undefined
          * For positive strides, begin() > end() implies size()==0
          * For negative strides, begin() < end() implies size()==0

A more general purpose container is the ``RAJA::ListSegment`` and typed ``TypedListSegment<T>; as before the former 
is constructed via :: 

   using RAJA::ListSegment = RAJA::TypedListSegment<RAJA::Index_type>


Lastly, the ``RAJA::StaticIndexSet<T>`` may be used to hold the various segments;
this enables assigning different execution policies for traversing through the segments and the values of the segments.
We refer the reader to the the following example ``example-gauss-sidel.cpp`` which illustrates the utility of the various segments.