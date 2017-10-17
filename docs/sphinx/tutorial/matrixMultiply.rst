.. _matrixMultiply::

===============
Matrix Multiply
===============

As a second example we consider matrix multiplication. Here two matrices, A, and B, of dimension
N x N are multiplied and the result is stored in a third matrix, C. 
Assuming that we have pointers to the data

.. literalinclude:: ../../../examples/example-matrix-multiply.cpp
                    :lines: 107-109

and with the aid of macros

.. literalinclude:: ../../../examples/example-matrix-multiply.cpp
                    :lines: 121-123

a C++ version of matrix multiplcation may be expressed as

.. literalinclude:: ../../../examples/example-matrix-multiply.cpp
                    :lines: 136-146

With minimal effort we can create an RAJA analog by modifying the existing block of code.
First, we can relive the need of macros by making use of ``RAJA::View``, which
simplifies multi-dimensional indexing (for more info see :ref:`ref-view`). 
                           
.. literalinclude:: ../../../examples/example-matrix-multiply.cpp
                    :lines: 155-157

Second, we can convert the outermost loop into a ``RAJA::forall`` loop

.. literalinclude:: ../../../examples/example-matrix-multiply.cpp
                    :lines: 167-180

yielding code which may be excuted with various backends.
In the case that the code will not be off loaded to a device, ``RAJA::forall`` loops
may be nested

.. literalinclude:: ../../../examples/example-matrix-multiply.cpp
                    :lines: 187-201  

Collapsing a series of nested loops may be done through the ``RAJA::forallN`` method.
Here it is necessary to use the ``RAJA::NestedPolicy`` (for more info see :ref:`ref-nested`) werein each
execution policy may be specified inside a ``RAJA::ExecList``. In the following example we pair the outer loop
with an OpenMP policy and the inner loop with a sequential policy.

.. literalinclude:: ../../../examples/example-matrix-multiply.cpp
                    :lines: 229-239

A full working version ``example-matrix-multiply.cpp`` may be found in the example folder.

