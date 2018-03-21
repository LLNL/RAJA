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

.. _matrixmultiply-label:

------------------------------------
Matrix Multiplication (Nested Loops)
------------------------------------

Key RAJA features shown in this example:

  * ``RAJA::forall`` loop traversal template 
  * RAJA execution policies
  * ``RAJA::View`` multi-dimensional data access
  * ``RAJA::kernel`` template for nested-loop execution
  * RAJA nested-loop interchange 


In this example, we multiply two square matrices 'A' and 'B' of dimension
N x N and store the result in matrix 'C'. To motivate the use of the 
``RAJA::View`` abstraction that we introduce here, we define the following 
macros to access the matrix entries in the C-version:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 56-58

Then, a typical C-style sequential matrix multiplication operation looks like:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 130-140

^^^^^^^^^^^^^^^^^^^^^
RAJA Forall Variants
^^^^^^^^^^^^^^^^^^^^^

In the RAJA variants of the matrix multiple example, we use two 
``RAJA::Range Segment`` objects to define the matrix row and column iteration 
spaces:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 163-165

We also employ ``RAJA::View`` objects, which allow us to access matrix 
entries in a multi-dimensional manner similar to the C-style version that 
uses macros. We create a two-dimensional N x N 'view' 
for each of the three matrices: 

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 153-155

Although we do not show such things in this example, RAJA views can be used to 
encapsulate a variety of different data layouts and access patterns, including 
permutations, stridings, etc. For more information about 
RAJA views, see :ref:`view-label`.

.. note:: The first few RAJA variants of the matrix multiplication operation
          do not illustrate recommended RAJA usage. They are included to 
          show *why they are not recommended* and to motivate the use of
          ``RAJA::kernel`` capabilities for nested loops.

In the first RAJA variant, we convert the outermost C-style 'row' loop to
use the ``RAJA::forall`` traversal method with a sequential execution policy:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 190-203

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
                    :lines: 225-238

Here, the lambda expression for the column loop body is contained within the 
lambda expression for the outer row loop. The outer loop execution policy
can be replaced with an OpenMP policy and the result will be the same as using 
an OpenMP policy in the previous version. 

However, changing the outer loop policy to a CUDA policy will not compile. 
This is by design in RAJA since nesting forall statements inside lambdas in 
this way has limited utility, is inflexible, and can hinder performance when 
compared to ``RAJA::kernel`` constructs, which we describe next. 

^^^^^^^^^^^^^^^^^
Nested-loop RAJA
^^^^^^^^^^^^^^^^^

In this section, we show how to recast the matrix-multiplication calculation 
by using ``RAJA::kernel`` nested-loop capabilities. There are some 
important differences worth noting between the use of the ``RAJA::kernel``
and ``RAJA::forall`` loop constructs. 

We first present a complete example, and then describe its key elements:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 270-290

Note that the ``RAJA::kernel`` template takes two arguments. Similar to the 
``RAJA::forall`` usage to this point, the first argument describes the
iteration space and the second argument is the lambda loop body. Here, the
index space is described as a *tuple* of ranges (created via the 
``RAJA::make_tuple`` method), one for each of the 'col' and 'row' loop 
The lambda has an index argument for each level in the loop nest. 

.. note :: The number and order of lambda arguments must match the number and
           order of ranges in the tuple for this to be correct.

The execution policy defined by the ``RAJA::KernelPolicy`` type specifies
a policy for each level in the loop nest via the nested 
``RAJA::statement::For`` types. Here, the row and column loops will both 
execute sequentially. The integer that appears as the first argument to each 
'For' template corresponds to the index of a range '(0, 1, 2, ...)' in the 
tuple and also to the associated lambda index argument; i.e., '0' is the 
'col' range, '1' is the 'row' range. The innermost type 
``RAJA::statement::Lambda<0>`` indicates that the first (and only!) lambda
expression passed to the ``RAJA::kernel`` method after the tuple of ranges
defines the inner loop body.  

The integer arguments in the 'For' statements are needed so that the levels 
in the loop nest can be 
permuted by reordering the policy arguments while the loop kernel remains 
the same. This is akin to how you would reorder C-style nested loops; i.e., 
reorder for-statements that loop over the indices for each loop nest level.
Here, the row policy is first (tuple index '1') so it is the outermost loop, 
and the col policy is next (tuple index '0') so it is the next loop, etc. We 
will demonstrate the reordering concept later in this example and discuss it in
more detail in the example :ref:`nestedreorder-label`.

If we wanted to execute the row loop using OpenMP multi-threaded parallelism 
and keep the column loop sequential as in earlier examples, the policy we 
would use is:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 302-309

To swap the loop nest ordering and keep the same execution policy on each loop,
we would use the following policy which swaps the ``RAJA::statement::For`` 
types. The inner loop is now the 'row' loop and is still parallel; 
the outer loop is now the 'col' loop and is still sequential:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
		    :lines: 339-346


The RAJA framework can also collapse nested loops in an OpenMP parallel region
using a ``RAJA::statement::Collapse`` type in the execution policy.
For example, the following policy will distribute the nested loop iterations 
differently and reduce the granularity of work done by each thread:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 374-380

The ``RAJA::ArgList`` type indicates which loops in the nest are to be 
collapsed and their nesting order within the collapse region.
The result of using this policy is essentially the same as using an OpenMP
'parallel for' directive on the outer loop with a 'collapse(2) clause. 

Note that collapsing loops via the OpenMP collapse clause may or may not 
improve performance. It depends on how the compiler creates
collapsed-loop indices using divide/mod operations and how well it can 
apply optimizations, such as dead-code-elimination. Note that there are no 
policies for the individual loop levels inside the OpenMP collapse policy. 

Lastly, we describe how to use ``RAJA::statement::CudaKernel`` types to 
collapse a loop nest into a single CUDA kernel and launch it with a particular 
thread-block decomposition. 

Here is a policy that will distribute the row indices across 
CUDA thread blocks and all column indices across threads in each
block:

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 416-425

This is equivalent to defining a CUDA kernel with the lambda body inside
it and defining row and column indices as::

  int row = blockIdx.x;
  int col = threadIdx.x;  

and launching the kernel with 'x' grid dimension N and 'x' blocksize N. 

The following policy will tile row and col indices across two-dimensional
CUDA thread blocks with 'x' and 'y' dimensions defined by the 'CUDA_BLOCK_SIZE'
parameter.

.. literalinclude:: ../../../../examples/ex4-matrix-multiply.cpp
                    :lines: 458-467

The file ``RAJA/examples/ex4-matrix-multiply.cpp``
contains the complete working example code. It also contains raw CUDA 
versions of the last RAJA CUDA example described here for comparison. 
