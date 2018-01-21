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

------------------------------------
Matrix Multiplication (Nested Loops)
------------------------------------

Key RAJA features shown in this example:

  * ``RAJA::forall`` loop traversal template 
  * RAJA execution policies
  * ``RAJA::nested::forall`` nested-loop traversal template 
  * ``RAJA::View`` multi-dimensional data access
  * RAJA nested-loop execution policies
  * RAJA nested loop interchange 


In this example, we multiply two square matrices 'A' and 'B' of dimension
N x N and store the result in matrix 'C'. To motivate the use of the 
``RAJA::View`` abstraction that we introduce here, we define the following 
macros to access the matrix entries in the C-version:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 58-60

Then, a typical C-style sequential matrix multiplication operation looks like:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 111-121

^^^^^^^^^^^^^^^^^^^^^
RAJA Forall Variants
^^^^^^^^^^^^^^^^^^^^^

In the RAJA variants of the matrix multiple example, we use two 
``RAJA::RangeSegment`` objects to define the matrix row and column iteration 
spaces:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 144-145

We also employ ``RAJA::View`` objects, which allow us to access matrix 
entries in a multi-dimensional manner similar to the C-style version that 
uses macros. We create a two-dimensional N x N 'view' 
for each of the three matrices: 

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 134-136

Although we do not show such things in this example, RAJA views can be used to 
encapsulate a variety of different data layouts and access patterns, including 
permutations, stridings, etc. For more information about 
RAJA views, see :ref:`view-label`.

In the first RAJA variant, we convert the outermost C-style 'row' loop to
use the ``RAJA::forall`` traversal method with a sequential execution policy:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 169-182

Here, the lambda expression for the loop body contains the 'col' and 'k'
loops.

Note that changing the execution policy to an OpenMP or CUDA policy enables the 
outer 'row' loop to run in parallel. When this is done, each thread will execute
the lambda expression body, which contains the 'col' and 'k' loops. Although 
this enables some parallelism, there is still more available. In a bit, we will
introduce RAJA nested loop abstractions and show that we can extract all 
available parallelism.

For the second RAJA variant, we nest a ``RAJA::forall`` traversal method
call for the 'column' loop inside the outer 'row' traversal:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 204-217

Here, the lambda expression for the column loop body is contained within the 
lambda expression for the outer row loop. The outer loop execution policy
can be replaced with an OpenMP policy and the result will be the same as using 
an OpenMP policy in the previous version. 

However, changing the outer loop policy to a CUDA policy will not compile. 
This is by design in RAJA since nesting forall statements inside lambdas in 
this way has limited utility, is inflexible, and can hinder performance when 
compared to ``RAJA::nested::forall`` constructs, which we describe next. 

.. note:: We recommend that RAJA users use ``RAJA::nested::forall`` constructs 
          for nested loops rather than nesting ``RAJA::forall`` statements 
          inside lambdas. 

^^^^^^^^^^^^^^^^^
Nested-loop RAJA
^^^^^^^^^^^^^^^^^

In this section, we show how to recast the matrix-multiplication calculation 
by using ``RAJA::nested::forall`` nested-loop capabilities. There are some 
important differences worth noting between nested-loop RAJA and the 
``RAJA::forall`` loop construct we have been using to this point. We
describe these differences in detail in the example :ref:`nestedreorder-label`.

We first present a complete example, and then describe its key elements:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 249-264

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
range '(0, 1, 2, ...)' in the tuple and also to the associated lambda index 
argument; i.e., '0' is the column range, '1' is the row range.  The integer 
arguments are needed so that the levels in the loop nest can be permuted by 
reordering the policy arguments while the loop kernel remains the same. 
This is akin to how you would reorder C-style nested loops; i.e., reorder
the for-statements that loop over the indices for each loop nest level.

Here, the row policy is first (tuple index '1') so it is the outermost loop, 
and the col policy is next (tuple index '0') so it is the next loop, etc. We 
will demonstrate the reordering concept later in this example and discuss it in
more detail in the example :ref:`nestedreorder-label`.

If we wanted to execute the row loop using OpenMP multi-threaded parallelism 
and keep the column loop sequential as in earlier examples, the policy we 
would use is:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 274-277

To swap the loop nest ordering and keep the same execution policy on each loop,
we would use the following policy which swaps the ``RAJA::nested::For`` 
arguments. The inner loop is now the 'row' loop and is still parallel; 
the outer loop is now the 'col' loop and is still sequential:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
		    :lines: 305-308


The RAJA framework can also collapse nested loops in an OpenMP parallel region
using a ``RAJA::nested::OmpParallelCollapse`` policy.
For example, the following policy will distribute the nested loop iterations 
differently and reduce the granularity of work done by each thread:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 334-338

The result of using this policy is essentially the same as using an OpenMP
'parallel for' directive on the outer loop with a 'collapse(2) clause. This
may or may not improve performance. This depends on how the compiler creates
collapsed-loop indices using divide/mod operations and how well it can 
apply optimizations, such as dead-code-elimination. Note that there are no 
policies for the individual loop levels inside the OpenMP collapse policy. 

Lastly, we describe how to use a ``RAJA::nested::CudaCollapse`` policy to 
collapse a loop nest into a single CUDA kernel and launch it with a particular 
thread-block decomposition. 

Here is a policy that will decompose all row iterations decomposed across the 
'y' thread block dimension and all column iterations decomposed across the 
'x' thread dimension:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 368-372

Another thread-block decomposition is achieved with the following:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 400-404

Here, the loop iterations are distributed over two-dimensional thread blocks
with x (col) and y (row) dimensions defined by the CUDA_BLOCK_SIZE_X and 
CUDA_BLOCK_SIZE_Y template arguments, respectively. In the example code, we 
use 16 x 16 thread blocks. 

The file ``RAJA/examples/ex4-matrix-multiply.cpp``
contains the complete working example code.
