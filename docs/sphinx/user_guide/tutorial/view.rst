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

.. _view-label:

---------------------------------------------
Batch Matrix-Multiply
---------------------------------------------

Key RAJA features shown in the following example:

  * ``RAJA::forall`` loop traversal template 
  * RAJA execution policies
  * ``RAJA::View`` multi-dimensional data access
  * ``RAJA::make_permuted_layout`` permutes data layouts

In this section, we present examples which carryout matrix multiplication for 
a batch of matrices each of dimension :math:`3 x 3`. More precisely we 
multiply :math:`A^e` :math:`\times` :math:`B^e` where :math:`e` enumerates a matrix in our batch, the 
result is then stored in :math:`C^e`.

The focus of this example is to illustrate how ``RAJA::layouts`` may be paired with the ``RAJA::Views``
to swap between different layouts for the entries in :math:`A^e` and :math:`B^e`. We introduce
the notation :math:`A^{e}_{rc}` to correspond to the entry in the row - :math:`r` , column - :math:`c`, 
of the matrix - :math:`A^e` . The two possible layouts we consider are the following

Layout 1: Assumes matrices are contiguous in memory thus enabling vectorizable operations

.. math::

   [A^{0}_{00}, A^{0}_{01}, A^{0}_{02}, \dots] .

Layout 2: Assumes matrix entries are grouped together for coalesced memory reads and writes.

.. math::

   [A^{0}_{00}, A^{1}_{00}, A^{2}_{00}, \dots] .

In this example we illustrate how RAJA simplifies swaping between layouts and thus simplifies architecture
architecture tuning.


^^^^^^^^^^
RAJA Timer
^^^^^^^^^^
To assess performance we mutiply the batch of matrices for a set number of iterations ``NITER``.
Performance is measured using the ``RAJA Timer``. Basic usage of the RAJA timer
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
Having introduced basic RAJA concepts in the previous secions, we dive into the RAJA forall method which multiplies a pair of matrices at each 
iteration. The snippet below unrolls the computation for each multiply. The iteration space spans between ``0 ... NMAT`` where ``NMAT`` is the 
total number of matrices to be multiplied. Furthermore, the arrays which hold the data ``A, B`` and ``C`` have been wrapped using a 
``RAJA::View`` to simplify dimensional indexing. 

.. literalinclude:: ../../../../examples/ex-batched-matrix-multiply.cpp
                    :lines: 149-163

The outermost index of the view corresponds to a column number, the second entry corresponds to a row number, and the inner most corresponds to 
the matrix in the batch number. In the construction of these views we take an additional step to construct a ``RAJA::layout`` which holds information about number of entries in each component and stride order. 

.. literalinclude:: ../../../../examples/ex-batched-matrix-multiply.cpp
                    :lines: 103-108

This particular version of the layout is made by the ``RAJA::make_permuted_layout`` method which takes in the number of entries in each component,
and specifies the ordering for the entries in the paranthesis operator. To construct a layout corresponding to the first layout 1, we use the following 
permutation ``RAJA::Perm<0, 1, 2>::get()`` which corresponds to the paranthesis operator ``Aview(e,r,c)`` indexing the array in the following manner::

            A[c + NCOLS*(r + NROWS*e)]

In particular ``c`` (the second entry) is marked with the shortest stride, then ``r`` (first entry), and finally ``e`` (the zeroth entry) as having the longest stride. Lastly, we template the ``RAJA::Views`` with ``RAJA::Layout`` templated on dimension, input type, and compoent with unit stride (in this case the second entry). By exposing unit stride the compiler may safely generate vector instructions. 

.. literalinclude:: ../../../../examples/ex-batched-matrix-multiply.cpp
                    :lines: 123-126

This particular permutation ``RAJA::Perm<1, 2, 0>`` permutes the ordering of the indices such that the shortest stride is now held by ``e``, then ``c``, and finally ``r``. This is equivalent to indexing the array in the following manner::

     Al2[e + NELEM*(r + NROWS*c)]. 


Run Time Comparison
-------------------
We report performance for various matrix batch numbers and their implications on performance. The experiments here are carried out 
on a Tesla K80 NVIDIA GPU, Intel Xeon CPU E5-2667 using the GNU 4.9 C++ compiler, and CUDA 9.::


   Number of matrices to be multiplied: 7000000 
   Matrix Multiplication with layout 1 on CPU run time : 0.121149 seconds
   Matrix Multiplication with layout 2 on CPU run time : 0.23192 seconds

   Matrix Multiplication with layout 1 on GPU run time : 0.134697 seconds
   Matrix Multiplication with layout 2 on GPU run time : 0.0175981 seconds

   Number of matrices to be multiplied: 12000000 
   Matrix Multiplication with layout 1 on CPU run time : 0.2205 seconds
   Matrix Multiplication with layout 2 on CPU run time : 0.530069 seconds

   Matrix Multiplication with layout 1 on GPU run time : 0.231031 seconds
   Matrix Multiplication with layout 2 on GPU run time : 0.0298856 seconds  

   Number of matrices to be multiplied: 24000000 
   Matrix Multiplication with layout 1 on CPU run time : 0.438874 seconds
   Matrix Multiplication with layout 2 on CPU run time : 1.74135 seconds

   Matrix Multiplication with layout 1 on GPU run time : 0.461516 seconds
   Matrix Multiplication with layout 2 on GPU run time : 0.0598168 seconds


As anticipated the first layout realizes better performance on the CPU while the second
layout realizes better performance on the GPU. 


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Three-dimensional Permuted Layouts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Lastly, we provide the analogues between C-style macros and RAJA permuted layouts for this example
  
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

