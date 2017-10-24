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

.. _forall::

======
Forall
======

The ``forall`` method is the building block for computations in RAJA, and
provides a way to write a loop so that it can be executed on a number of
different programming model backends. Backend selection is controlled through
policies, documented in :doc:`policies`. Its basic usage is as follows

.. code-block:: cpp

  RAJA::forall<exec_policy>(iter_space I, [=] (index_type i)) {
    //body
  });

The construction of a ``RAJA::forall`` requires

* Capture type - [=] or [&]

* exec_policy  - How the traversal occurs

* iter_space   - An iteration space for the RAJA loop (any random access container)

* index_type   - Type of values contained in the iteration space

* lambda       - The body of the loop



