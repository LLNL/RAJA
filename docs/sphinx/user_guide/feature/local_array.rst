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

.. _local_array-label:

===========
Local Array
===========

RAJA kernel enables developers to express nested and non-perfectly nested loops, in this section
we introduce the ``RAJA::LocalArray`` object, which enables developers to construct arrays between
subsequent loops. For clarity, constructing an array between consecutive loops in C++ would look like the
following::

           for(int i = 0; i < 50; ++i) {
           
             double array[10];

             for(int j = 0; j < 10; ++j) {
               array[j] = 3*j
             }

             for(int j = 0; j < 10; ++j) {
               std::cout<< array[j] <<std::endl;
             }

           }


Similary, RAJA enables the construction of arrays within the scope of a loop and have the array 
accessible to subsequent loops. As an example, we will illustrate how to construct and initalize
a RAJA local array. 

A RAJA local array is defined outside a RAJA kernel by specifying the type of the data and 
the dimensions of the array as template arguments. 
In the example below we construct an array of size 5 math:\times 10::

    RAJA::LocalArray<double, RAJA::Sizes<5,10> > A; 

.. note:: RAJA Local arrays support arbiratry dimensions and sizes.

Although the object has been constructed, its memory has not be initalized. Memory intialization for a 
RAJA local array is done through the ``InitLocalMem`` statement in a kernel policy. 