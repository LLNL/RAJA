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

Loop counters and iteration spaces are fundamental to C/C++ loops. In this section we introduce RAJA specific index types 
and iteration space which are designed to encapsulate common programming patterns.

.. note:: * All segment objects and the ``Index_type`` are found in the namespace RAJA

-------
Indices
-------

Although RAJA may utilize any loop counter type, the recommended loop counter is the ``RAJA::Index_type``. The ``RAJA::Index_type`` 
is a 64-bit loop counter which makes it easy for the compiler to carryout optimizations.

-------------
RAJA Segments
-------------

RAJA introduces a family of segments which serve as containers for iterations spaces. 
The fundamental segment types are ``RAJA::RangeSegment``, and ``RAJA::TypedRangeSegment``; the former is constructed is 
an alias for a ``RAJA::TypedRangeSegment<RAJA::Index_type>``. Basic usage is as follows ::

   //  Generates a contiguous sequence of numbers by the [start, stop) interval specified 
   RAJA::TypedRangeSegment<T>(T start, T stop)  
   //or                                                           
   RAJA::RangeSegment loopBounds(RAJA::Index_type start, RAJA::Index_type stop)
    

Under the RAJA programming model, the purpose of these containers is to generate a contiguous sequence of numbers and more fundamentally,
to serve as the iteration space for loops::

   RAJA::forall<exec_policy>(TypedRangeSegment<T>(T begin,T end), [=] (RAJA::Index_type i)) {
     //loop boody
   }

.. note:: * TypedRangeStrideSegment::iterator is a random access iterator
          * TypedRangeStrideSegment allows for positive or negative strides, but a stride of zero is undefined
          * For positive strides, begin() > end() implies size()==0
          * For negative strides, begin() < end() implies size()==0

A more general purpose container is the ``RAJA::ListSegment`` and typed ``RAJA::TypedListSegment<T>``; as before the ``RAJA::ListSegment`` is an alias for a 
``RAJA::TypedRangeSegment<RAJA::Index_type>``. Both of these containers store non-contiguous indices. Basic usage of the ``RAJA::ListSegment``::

    //*Aptr points to an array of values to traverse and Alen is number of elements.
    RAJA::TypedListSegment<T>(T *Aptr, T Alen)  
    //or
    RAJA::ListSegment(RAJA::Index_type *Aptr,RAJA::Index_type Alen)


Lastly, the ``RAJA::StaticIndexSet<T>`` may be used to hold different instances of segments. The utility of the ``RAJA::StaticIndexSet<T>`` is demonstrated in the Red-Black Gauss-Sidel algorithm wherein parallism may be exposed by decomposing the algorithm into two sweeps.
We refer the reader to the example ``example-gauss-sidel.cpp`` for further detail. 
