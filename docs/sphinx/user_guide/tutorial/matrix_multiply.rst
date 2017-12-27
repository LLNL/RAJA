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

.. _matrixmultiply-label:

-----------------------
Matrix Multiplication
-----------------------

Key RAJA features shown in this example:

  * ``RAJA::forall`` loop traversal template 
  * RAJA execution policies
  * ``RAJA::nested::forall`` nested-loop traversal template 
  * ``RAJA::View`` multi-dimensional data access
  * RAJA nested-loop execution policies
  * RAJA nested loop interchange 


In this example, we multiply two square matrices 'A' and 'B' of dimension
N x N and store the result in matrix 'C'. To motivate the use of the 
``RAJA::View`` abstraction, we define the following macros to access the 
matrix entries in the C-version:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 57-59

Then, a typical C-style sequential matrix multiplication operation looks like:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 110-118

^^^^^^^^^^^^^^^^^^^^^
Initial RAJA Variants
^^^^^^^^^^^^^^^^^^^^^

In the RAJA variants of the matrix multiple example, we use two 
``RAJA::RangeSegment`` objects to define the matrix row and column iteration 
spaces:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 143-144

We also use ``RAJA::View`` objects, which allow us to access matrix 
entries in a multi-dimensional manner similar to the C-style version but 
without the need for macros. Here, we create a two-dimensional N x N 'view' 
for each of the three matrices: 

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 133-135

Although we do not show such things here, RAJA Views can be used to encapsulate
a variety of different data properties and access patterns, including 
different layouts, permutations, stridings, etc. For more information about 
RAJA views, see :ref:`view-label`.

In the first RAJA variant, we convert the outermost C-style 'row' loop to
use the ``RAJA::forall`` traversal method with a sequential execution policy:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 161-174

Here, the lambda expression for the loop body contains the 'col' and 'k'
loops.

Note that changing the execution policy to a RAJA OpenMP policy, for example, 
would enable the outer loop to run in parallel using CPU multi-threading. 
Later, when we introduce RAJA nested loop abstractions, we will show another 
way to do this.

For the second RAJA variant, we nest a ``RAJA::forall`` traversal method
call for the 'column' loop inside the outer 'row' traversal:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 190-201

Here, the lambda expression for the column loop body is contained within the 
lambda expression for the outer row loop. When the code will not be run on a 
GPU using CUDA, lambda expressions may be nested in this way. Enabling such
a nesting for CUDA device lambdas is a feature we have been discussing with
the NVIDIA compiler team.

^^^^^^^^^^^^^^^^^
Nested-loop RAJA
^^^^^^^^^^^^^^^^^

In this section, we recast the matrix-multiplication operation using
``RAJA::nested::forall`` nested-loop capabilities. Before we begin,
there are some important differences worth noting between nested-loop RAJA 
and the ``RAJA::forall`` loop construct we have been using to this point:

  * A range and lambda index argument are required for each level in a 
    loop nest. In this example, we focus on doubly-nested loops.

  * A range for each loop nest level is specified in a RAJA tuple object.
    The order of ranges in the tuple must match the order of arguments to the
    lambda for this to be correct, in general. RAJA provides strongly-typed
    indices to help with this. They are explained in a later example
    :ref:`nestedreorder-label`.

  * An execution policy is required for each level in the loop nest. These
    are specified in the ``RAJA::nested::Policy`` type.

  * The loop nest ordering is specified in the nested execution policy -- 
    the first 'For' policy is the outermost loop, the second 'For' policy  
    is the loop nested inside the outermost loop, and so on.  

We first present the complete example, and then describe its key elements:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 236-250

The ``RAJA::nested::forall`` template takes three arguments: a nested policy,
a tuple of ranges, one for each level in the loop nest, and the lambda 
loop body. The lambda has an index argument for each level in the
loop nest. 

.. note :: The number and order of lambda arguments must match the number and
           order of ranges in the tuple for this to be correct.

The execution policy for each level in the loop nest is defined by the 
``RAJA::nested::For`` arguments in the nested policy type. Here, the row
and column loops will both execute sequentially. The integer that appears
as the first argument to each 'For' template corresponds to the index of a
range in the tuple; i.e., '0' is the column range, '1' is the row range.
The reason that the integer arguments are needed is that the levels in the
loop nest can be permuted by reordering the policy arguments. Here, the row
policy is first (tuple index '0') so it is the outermost loop, and the row 
policy is next (tuple index '1') so it is the next loop, etc. We will 
demonstrate the reordering concept later in this example and discuss it in
more detail in the example :ref:`nestedreorder-label`.

If we wanted to execute the row loop using OpenMP multi-threaded parallelism 
and keep the column loop sequential, the policy we would use is:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 260-262

To swap the loop nest ordering and keep the same execution policy on each loop,
we would use this policy:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 290-292

Lastly, we discuss a policy that collapses the loop nest into a single CUDA
kernel and launches it with a particular thread block decomposition:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 322-325

Here, the loop iterations are distributed over two-dimensional thread blocks
with x and y dimensions defined by the CUDA_BLOCK_SIZE_X and CUDA_BLOCK_SIZE_Y 
template arguments, respectively. In the example code, we use 16 x 16 thread
blocks. The row iterations (tuple index '1') are distributed over the 'y'
thread block dimension and the column iterations (tuple index '0') are 
distributed over the 'x' thread block dimension, as indicated by the execution
policies.

The file ``RAJA/examples/ex4-matrix-multiply.cap`` 
contains the complete working example code.
