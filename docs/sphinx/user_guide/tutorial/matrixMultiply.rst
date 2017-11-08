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

.. _matrixmult-label:

-----------------------
Matrix Multiplication
-----------------------

Main RAJA features discussed:

  * ``RAJA::View`` multi-dimensional data access
  * ``RAJA::forall`` loop traversal template 
  * ``RAJA::forallN`` nested-loop traversal template 
  * RAJA execution policies, including nested-loop policies


In this example, we multiply two matrices 'A' and 'B' of dimension
N x N and store the result in matrix 'C'. To simplify the example, we
use the following macros to access the matrix entries:

.. literalinclude:: ../../../../examples/example-matrix-multiply.cpp
                    :lines: 107-109

Also, we use a 'memory manager' to simplify allocation of CPU vs. GPU 
memory depending on how we wish to run the example.  

.. literalinclude:: ../../../../examples/example-matrix-multiply.cpp
                    :lines: 121-123

Assuming we have initialized the arrays holding the data for the matrices,
a typical C-style nested loop to multiple the matrices is: 

.. literalinclude:: ../../../../examples/example-matrix-multiply.cpp
                    :lines: 136-146

^^^^^^^^^^^^^^^^^^
Converting to RAJA
^^^^^^^^^^^^^^^^^^

In the RAJA version of the example code, we use the ``RAJA::View`` capability,
which allows us to access matrix entries in a multi-dimensional manner similar 
to the C-style version but without the need for macros. Here, we create a
two-dimensional N x N 'view' for each matrix: 

.. literalinclude:: ../../../../examples/example-matrix-multiply.cpp
                    :lines: 155-157

For more information about RAJA views, see :ref:`view-label`.
                           
First, we can convert the outermost loop to use the ``RAJA::forall`` traversal
method with a sequential execution policy:

.. literalinclude:: ../../../../examples/example-matrix-multiply.cpp
                    :lines: 167-180

Here, the RAJA loop iteration space is defined by a RAJA RangeSegment object:

.. literalinclude:: ../../../../examples/example-matrix-multiply.cpp
                    :lines: 163

Changing the execution policy to a RAJA OpenMP policy, for example, would 
enable the outer to run in parallel using CPU multi-threading. See the 
``RAJA::forallN`` version of the example below for another way to do this.

When the code will not be run on a GPU, ``RAJA::forall`` loops may be nested
as in the following:

.. literalinclude:: ../../../../examples/example-matrix-multiply.cpp
                    :lines: 187-201  

^^^^^^^^^^^^^^^^^
Nested-loop RAJA
^^^^^^^^^^^^^^^^^

Here, we recast the matrix-multiplication using the ``RAJA::forallN`` 
nested-loop capability.  

Using ``RAJA::forallN``, requires that the execution policy for each loop in 
the nest be described using ``RAJA::NestedPolicy`` and a ``RAJA::ExecList``. 
Here, the outer loop has an OpenMP 'parallel for' execution policy and the
loop nested inside uses a sequential policy:

.. literalinclude:: ../../../../examples/example-matrix-multiply.cpp
                    :lines: 229-240

For more information about RAJA nested-loop functionality, 
see :ref:`forall-label`.

.. note:: The order of execution policies specified in a RAJA nested-loop
          policy is outermost to innermost, left to right. 

The file ``RAJA/examples/example-matrix-multiply.cpp`` 
contains the complete working example code.
