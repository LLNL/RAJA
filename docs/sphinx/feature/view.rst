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

.. _view::
.. _ref-view:
 
===============
View and Layout
===============

Creating accesors for multidimensional arrays 

.. code-block:: cpp

   double *A = new double [N*N]

can be done in various ways, a classic manner is through macros

.. code-block:: cpp
   
   #define A(x2, x1) A[x1 + N * x2]

RAJA simplifies multi-dimensional indexing by introducing the ``RAJA::View``. The basic usage is 
as follows 

.. code-block:: cpp

   RAJA::View<double, RAJA::Layout<DIM>> Aview(A, N1, ..., Nn);

Here the ``RAJA::View`` is templated on a type (ex. double, float, int, ...), and ``N1, ... , Nn``
identifies the stride in each dimension. 

The ``RAJA::Layout<DIM>`` encapsulates the number of dimensions , ``DIM`` , the ``RAJA::View`` will have.
Accesing entries may then be done through the following accessor

.. code-block:: cpp

   Aview(x2,x1)

Basic usage is illustrated in ``examples-matrix-multiply.cpp``.



