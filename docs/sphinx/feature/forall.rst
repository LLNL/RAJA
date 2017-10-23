.. _forall::

======
Forall
======

The ``forall`` method is the building block for computations in RAJA, and
provides a way to write a loop so that it can be executed on a number of
different programming model backends. Backend selection is controlled through
policies, documented in :doc:`policies`_

The ``RAJA::forall`` method is an abstraction of the standard C for loop. 
It is templated on an execution policy and takes an iteration space and a lambda capturing the loop body as arguments.

.. code-block:: cpp

  RAJA::forall<exec_policy>(iter_space I, [=] (index_type i)) {
    //body
  });

The construction of a ``RAJA::forall`` requires

1. Capture type - [=] or [&]

2. exec_policy  - How the traversal occurs

3. iter_space   - An iteration space for the RAJA loop (any random access container)

4. index_type   - Type of values contained in the iteration space

5. lambda       - The body of the loop



