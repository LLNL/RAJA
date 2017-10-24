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

.. _nested::
.. _ref-nested:

=============
Nested Forall
=============

In situations were loops are nested

.. code-block:: cpp

  for(int i1 = 0; i1 < N; ++i1){
    for(int i2 = 0; i2 < N; ++i2){
              ...
    \\body
              
    }
  }

RAJA provides an abstraction which may be combined with various execution policies. The ``RAJA::ForallN`` loop is templated 
on up to N execution policies and expects an iteration space for each execution policy and a lambda with an argument
for each iteration space.

.. code-block:: cpp

  RAJA::forallN<
    RAJA::NestedPolicy<exec_policy1, ... , exec_policyN> >(
      iter_space I1, ..., iter_space IN, [=] (index_type i1, ..., index_type iN) {

      });


Here the developer is supplies the following

* Capture type - [=] or [&]

* exec_policy  - How the traversal occurs

* iter_space   - An iteration space for the RAJA loop (any random access container)

* index_type   - Type of values contained in the iteration space

* lambda       - The body of the loop