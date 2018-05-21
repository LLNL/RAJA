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

.. _permuted-layout-label:

---------------------------------------------
Batch Matrix-Multiply 
---------------------------------------------

Key RAJA features shown in the following example:

  * ``RAJA::forall`` loop traversal template 
  * RAJA execution policies
  * ``RAJA::View`` multi-dimensional data access
  * ``RAJA::make_permuted_layout`` permutes how data is accessed through the view paranthesis operator

In this section, we present examples which carryout matrix multiplication for 
a batch of matrices each of dimension :math:`3 x 3`. More precisely we 
multiply :math:`A^e` :math:`\times` :math:`B^e` where :math:`e` enumerates a matrix in our batch. The 
result is then stored in :math:`C^e`.

The focus of this example is to illustrate how ``RAJA::layouts`` may be paired with ``RAJA::Views``
to swap between different layouts for the entries in :math:`A^e` and :math:`B^e`. We introduce
the notation :math:`A^{e}_{rc}` to correspond to the entry in the row - :math:`r` , and column - :math:`c`
of the matrix - :math:`A^e`. Futhermore we alternate between two potential layouts to study their impact on
performance.

Layout 1: Assumes matrices are contiguous in memory thus enabling vectorizable operations

.. math::

   [A^{0}_{00}, A^{0}_{01}, A^{0}_{02}, \dots] .

Layout 2: Assumes matrix entries are grouped by matrix number for coalesced memory reads and writes.

.. math::

   [A^{0}_{00}, A^{1}_{00}, A^{2}_{00}, \dots] .

^^^^^^^^^^
RAJA Timer
^^^^^^^^^^
To evaluate performance between layouts we mutiply a batch of matrices for a set number of iterations
and compare the minimum run time. Performance is measured by using the ``RAJA timer``. Basic usage of the RAJA timer
is illustrated below::

    auto timer = RAJA::Timer(); 
    
    timer.start(); //start recording time

    //run kernel

    timer.stop();  //stop recording time
    
    auto elapsedTime = timer.elapsed(); //report total amount elapsed time
    timer.reset()  //time is aggregated and the timer must be reset


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Batched Matrix Multiplication with RAJA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Having introduced basic RAJA concepts in the previous sections, we dive into the RAJA forall method which multiplies a pair of matrices at each 
iteration. The snippet below unrolls the computation for each matrix multiply. The iteration space spans between :math:`0 ... NMAT` where :math:`NMAT` is the 
total number of matrices to be multiplied. Furthermore, the arrays which hold matrix data :math:`A`, :math:`B` and :math:`C` have been wrapped using a 
``RAJA::View`` to simplify dimensional indexing. 

.. literalinclude:: ../../../../examples/ex-batched-matrix-multiply.cpp
                    :lines: 153-167

The outermost index of the view corresponds to a column number, the second entry corresponds to a row number, and the inner most index corresponds to 
the matrix number in the batch. For the construction of the views we take an additional step to construct  a``RAJA::layout`` which will hold information about
number of entries in each component and stride order. 

.. literalinclude:: ../../../../examples/ex-batched-matrix-multiply.cpp
                    :lines: 103-108

This particular version of the layout is made by the ``RAJA::make_permuted_layout`` method. The method takes in the number of entries in each component
and a desired ordering for the indices in the paranthesis operator. To construct a ``RAJA::layout`` corresponding to the first layout
we use the following permutation ``RAJA::Perm<0, 1, 2>::get()``. The permutation corresponds to the paranthesis operator ``Aview(e,r,c)`` indexing 
the array in the following manner::

            A[c + NCOLS*(r + NROWS*e)]

In particular ``c`` (the second entry in the view) is denoted to have the shortest stride by the layout, ``r`` (the first entry) has the second shortest stride, and ``e`` (the zeroth entry) is marked to have the longest stride. Lastly, we template the ``RAJA::Layout`` inside the ``RAJA::View`` on dimension, input type, and component with unit stride (in this case the second entry). By exposing unit stride the compiler may identify opportunities to generate
vector instructions. The ``RAJA::layout`` corresponding to the second desired layout may be constructed in the following manner

.. literalinclude:: ../../../../examples/ex-batched-matrix-multiply.cpp
                    :lines: 123-126

The permutation ``RAJA::Perm<1, 2, 0>`` permutes the ordering of the indices such that the shortest stride is now held by ``e``, then ``c``, and finally ``r``. This is equivalent to indexing the array in the following manner::

     A[e + NELEM*(r + NROWS*c)]. 


Run Time Comparison
-------------------
We report run time for various matrix-multiplication batch numbers. The experiments here are carried out 
on a Tesla K80 NVIDIA GPU, Intel Xeon CPU E5-2667 using the GNU 4.9 C++ compiler and CUDA 9. ::


   Number of matrices to be multiplied: 7000000 

   Matrix Multiplication with layout 1 on CPU with loop policy run time : 0.120773 seconds
   Matrix Multiplication with layout 2 on CPU with loop policy run time : 0.246547 seconds
   
   Matrix Multiplication with layout 1 on CPU with OpenMP parallel policy run time : 0.0380581 seconds
   Matrix Multiplication with layout 2 on CPU with OpenMP parallel policy run time : 0.0383805 seconds
   
   Matrix Multiplication with layout 1 on GPU with Cuda policy run time : 0.134659 seconds
   Matrix Multiplication with layout 2 on GPU with Cuda policy run time : 0.0176018 seconds

   Number of matrices to be multiplied: 12000000
   
   Matrix Multiplication with layout 1 on CPU with loop policy run time : 0.218981 seconds
   Matrix Multiplication with layout 2 on CPU with loop policy run time : 0.529489 seconds

   Matrix Multiplication with layout 1 on CPU with OpenMP parallel policy run time : 0.0652577 seconds
   Matrix Multiplication with layout 2 on CPU with OpenMP parallel policy run time : 0.0652921 seconds

   Matrix Multiplication with layout 1 on GPU with Cuda policy run time : 0.231011 seconds
   Matrix Multiplication with layout 2 on GPU with Cuda policy run time : 0.0298882 seconds

Here we observe that the first layout realizes better performance under a ``RAJA::loop_exec`` policy while the second
layout relizes better performance on the GPU. The file ``RAJA/examples/ex-offset.cpp`` contains the complete working example code.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Three-dimensional Permuted Layouts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For completeness, we provide the analogues between C-style macros and RAJA permuted layouts for this example::
  
  #define A(elem, row, col) A[elem + Nelem*(col  + row*NCOLS)]
  auto layout = RAJA::make_permuted_layout({{Nelem,NROWS, NCOLS}}, RAJA::as_array<RAJA::Perm<1,2,0> >::get() );
       
  #define A(elem, row, col) A[elem + Nelem*(row + NROWS*col)
  auto layout = RAJA::make_permuted_layout({{Nelem, NROWS, NCOLS}}, RAJA::as_array<RAJA::Perm<2,1,0> >::get() );

  #define A(elem, row, col) A[col + NCOLS*(row + NROWS*elem)]
  auto layout = RAJA::make_permuted_layout({{Nelem, NROWS, NCOLS}}, RAJA::as_array<RAJA::Perm<0,1,2> >::get() );

  #define A(elem, row, col) A[col + NCOLS*(elem + Nelem*row)]
  auto layout = RAJA::make_permuted_layout({{Nelem, NROWS, NCOLS}}, RAJA::as_array<RAJA::Perm<1,0,2> >::get() );

  #define A(elem, row, col) A[row + NROWS*(elem + Nelem*col)]
  auto layout = RAJA::make_permuted_layout({{Nelem, NROWS, NCOLS}}, RAJA::as_array<RAJA::Perm<2,0,1> >::get() );

  #define A(elem, row, col) A[row + NROWS*(col + NCOLS*elem)]
  auto layout = RAJA::make_permuted_layout({{Nelem, NROWS, NCOLS}}, RAJA::as_array<RAJA::Perm<0,2,1> >::get() );

