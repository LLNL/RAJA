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

.. _traits-label:

========
Atomics
=======

In operations where we wish to avoid race conditions in specific memory locations, RAJA
introduces templated atomic methods. The list of atomic operations and list of polcies are found below. 

-----------------
Atomic Operations
-----------------

* ``RAJA::atomic::atomicAdd<AtomicPolicy>(* T acc, T value)``  - Add acc by value

* ``RAJA::atomic::atomicSum<AtomicPolicy>(* T acc, T value)``  - Subtracts acc by value

* ``RAJA::atomic::atomicMin<AtomicPolicy>(* T acc, T value)``  - Returns the maximum to acc

* ``RAJA::atomic::atomicMax<AtomicPolicy>(* T acc, T value)``  - Returns the minimum to acc

* ``RAJA::atomic::atomicInc<AtomicPolicy>(* T acc)``  - Increments acc by 1

* ``RAJA::atomic::atomicDec<AtomicPolicy>(* T acc)``  - Decreases acc value by 1 

Remark: The left most argument is assumed to be the pointer to the memory location. 

---------------
Atomic Policies
---------------

* ``seq_atomic``     - Unprotected operation.

* ``auto_atomic``    - Attempts to determine the correct policy. 

* ``buildin_atomic`` - Uses compiler specific atomics

* ``omp_atomic``     - Uses omp atomic pragma

* ``cuda_atomic``    - Uses cuda specicic atomic

An example of basic usage is found in ``example-atomic-pi.cpp``. 