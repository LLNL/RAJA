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

Key RAJA features shown in the following examples:

  * ``RAJA::forall`` loop traversal template 
  * RAJA execution policies
  * ``RAJA::View`` multi-dimensional data access
  * ``RAJA::kernel`` template for nested-loop execution
  * Basic RAJA nested-loop interchange 


In this section, we present examples showing multiplication of two square 
matrices 'A' and 'B' of dimension N x N where we store the result in matrix 'C'.
To motivate the use of the ``RAJA::View`` abstraction that we introduce here, 
we define the following macros to access the matrix entries in the C-version:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
                    :lines: 56-58

Then, a typical C-style sequential matrix multiplication operation looks like:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
                    :lines: 130-140

^^^^^^^^^^^^^^^^^^^^^
RAJA::forall Variants
^^^^^^^^^^^^^^^^^^^^^

In the RAJA variants of the matrix multiple example, we use two 
``RAJA::Range Segment`` objects to define the matrix row and column and dot
product iteration spaces:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
                    :lines: 163-165

We also employ ``RAJA::View`` objects, which allow us to access matrix 
entries in a multi-dimensional manner similar to the C-style version that 
uses macros. We create a two-dimensional N x N 'view' 
for each of the three matrices: 

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
                    :lines: 153-155

Although we do not show such things in this example, RAJA views can be used to 
encapsulate a variety of different data layouts and access patterns, including 
permutations, stridings, etc. For more information about 
RAJA views, see :ref:`view-label`.

.. note:: The first few RAJA variants of the matrix multiplication operation
          show RAJA usage that **we do not recommend**; we noted some
          rationale behind this in :ref:`loop_elements-kernel-label`. These
          versions using ``RAJA::forall`` are included to show limitations of 
          the approach and to motivate the use of ``RAJA::kernel`` constructs 
          for nested loops.

In the first RAJA variant, we convert the outermost C-style 'row' loop to
use the ``RAJA::forall`` traversal method with a sequential execution policy:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
                    :lines: 191-204

Here, the lambda expression for the loop body contains the 'col' and 'k' loops.

Note that changing the execution policy to an OpenMP or CUDA policy, for 
example, enables the outer 'row' loop to run in parallel. When this is done, 
each thread executes the lambda expression body, which contains the 'col' 
and 'k' loops. Although this enables some parallelism, there is still more 
available. In a bit, we will show how to use the ``RAJA::kernel`` interface
so that we can extract all available parallelism.

For the second RAJA variant, we nest a ``RAJA::forall`` traversal method
call for the 'column' loop inside the outer 'row' traversal:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
                    :lines: 226-239

Here, the innermost lambda expression contains the inner 'k' loop. The 'col' 
loop is executed via the ``RAJA::forall`` method call within the 
lambda expression for the outer 'row' loop. Note that we can replace either
forall execution policy with an OpenMP execution policy to parallelize either
the 'row' or 'col' loop. For examples, if we use an OpenMP execution policy 
on the outer 'row' loop, the result will be the same as using an OpenMP policy 
in the ``RAJA::forall`` statement in the previous example. 

We do not recommended to use a parallel execution policy for both loops in 
this example as the results may not be what is expected and RAJA provides 
better mechanisms for parallelizing nested loops. Also, changing the outer 
loop policy to a CUDA policy will not compile. This is by design in RAJA 
since nesting forall statements inside lambdas in this way has limited utility, 
is inflexible, and can hinder performance when compared to ``RAJA::kernel`` 
constructs, which we describe next. 

.. _matmultkernel-label:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Basic RAJA::kernel Variants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this section, we show how to cast the matrix-multiplication operation 
using ``RAJA::kernel`` nested-loop capabilities, which were introduced in
:ref:`loop_elements-kernel-label`. We first present a complete example, and
then describe its key elements, noting important differences between
the ``RAJA::kernel`` and ``RAJA::forall`` loop constructs. 

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
                    :lines: 276-295

Here, we use ``RAJA::kernel`` to express the outer 'row' and 'col' loops; 
the inner 'k' loop is represented by the lambda expression. Note that 
the ``RAJA::kernel`` template takes two arguments. Similar to ``RAJA::forall``,
the first argument describes the iteration space and the second argument is 
the lambda loop body. Unlike ``RAJA::forall``, the iteration space is defined 
as a *tuple* of ranges (created via the ``RAJA::make_tuple`` method), one for 
the 'col' loop and one for the 'row' loop. Also, the lambda has an index 
argument for each level in the loop nest. 

.. note :: The number and order of lambda arguments must match the number and
           order of the elements in the tuple for this to be correct.

The main different between ``RAJA::forall`` and ``RAJA::kernel`` is in the
execution policy template parameter. The execution policy defined by the 
``RAJA::KernelPolicy`` type used here specifies a policy for each level in 
the loop nest via nested ``RAJA::statement::For`` types. Here, the row and 
column loops will both execute sequentially. The integer that appears as the 
first template parameter to each 'For' statement corresponds to the index of 
a range '(0, 1, ...)' in the tuple and also to the associated index argument 
to the lambda. Here, '0' is the 'col' range and '1' is the 'row' range because 
that is the order those ranges appear in the tuple. The innermost type 
``RAJA::statement::Lambda<0>`` indicates that the first lambda expression
(the only one in this case!) argument passed to the ``RAJA::kernel`` method 
after the index space tuple will be invoked inside the nested loops.

The integer arguments in the 'For' statements are needed so that the levels 
in the loop nest can be permuted by reordering the policy arguments while the 
loop kernel code remains the same. As noted in 
:ref:`loop_elements-kernel-label`, this is analogous to how one would reorder 
C-style nested loops; i.e., reorder for-statements for each loop nest level. 

If we want to execute the row loop using OpenMP multi-threaded parallelism 
and keep the column loop sequential, the policy we would use is:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
                    :lines: 307-314

To swap the loop nest ordering and keep the same execution policy on each loop,
we would use the following policy, which swaps the ``RAJA::statement::For`` 
types. The inner loop is now the 'row' loop and is run in parallel; 
the outer loop is now the 'col' loop and is still sequential:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
		    :lines: 343-350

. note:: It is important to note that these kernel transformations, and others,
         can be done by switching the ``RAJA::KernelPolicy`` type with no
         changes to the loop kernel code.

In :ref:`nestedreorder-label`, we provide a more detailed discussion of the
mechanics of loop nest reordering. Next, we show other variations of the 
matrix multiplication kernel that illustrate other ``RAJA::kernel`` features. 

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
More Complex RAJA::kernel Variants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The set of matrix multiplication kernel variations in this section use
execution policies to express the outer row and col loops as well as the 
inner dot product loop using the RAJA kernel interface. They show some more 
complex policy examples and use additional RAJA kernel features.

The first example uses sequential execution for all loops:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
                    :lines: 512-545

Here, we use the ``RAJA::kernel_param`` method to execute the kernel since
we pass in an object to hold a scalar thread-local variable for the dot product 
sum as the second argument. It is passed as a single-valued tuple object since,
in general, we can pass in any number of variables to the method. The first
argument is a tuple of iteration spaces for the loop levels. The remaining
arguments include a sequence of lambda expressions representing different
parts of the inner loop body. We use three lambda expressions that: initialize 
the dot product variable (lambda 0), define the 'k' inner loop row-col dot 
product operations (lambda 1), and store the computed row-col dot product 
in the proper location in the result matrix (lambda 2). Note that all lambdas
take the same arguments in the same order, which is required for the kernel
to be well-formed. In addition to the loop index variables, we pass the scalar 
dot product variable into each lambda. This enables the same variable to be 
by all three lambdas. However, also note that not all lambda expressions use
all three index variables.

The execution policy type passed to the ``RAJA::kernel_param`` method as a 
template parameter describes how the nested loops and lambda expressions are
assembled to for the complete kernel.

The RAJA framework can also collapse nested loops in an OpenMP parallel region
using a ``RAJA::statement::Collapse`` type in the execution policy. This
allows multiple levels in a loop nest to be parallelized using OpenMP 
directives. For example, the following policy will collapse the two outer loops:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
                    :lines: 558-568

The ``RAJA::ArgList`` type indicates which loops in the nest are to be 
collapsed and their nesting order within the collapse region. Note that there 
are no policies for the individual loop levels inside the OpenMP collapse 
policy. The result of using this policy is essentially the same as using an 
OpenMP 'parallel for' directive with a 'collapse(2) clause on the outer loop 
in the original C-style variant. This will distribute the iterations for 
both loop levels across OpenMP threads, increasing the granularity of work 
done by each thread. This may or may not improve performance. It depends on how 
the compiler creates collapsed-loop indices using divide/mod operations and 
how well it can apply optimizations, such as dead-code-elimination. 

Lastly, we describe how to use ``RAJA::statement::CudaKernel`` types to 
generate a CUDA kernels launched with particular thread-block decompositions.
Note that the policies are different, but that the kernels themselves are
identical to the sequential and OpenMP variants above.

Here is a policy that will distribute the row indices across 
CUDA thread blocks and all column indices across threads in each
block:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
                    :lines: 604-617

This is equivalent to defining a CUDA kernel with the lambda body inside
it and defining row and column indices as::

  int row = blockIdx.x;
  int col = threadIdx.x;  

and launching the kernel with 'x' grid dimension N and 'x' blocksize N. 

The following policy will tile row and col indices across two-dimensional
CUDA thread blocks with 'x' and 'y' dimensions defined by the 'CUDA_BLOCK_SIZE'
parameter.

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
                    :lines: 650-663

The file ``RAJA/examples/tut_matrix-multiply.cpp``
contains the complete working example code for the examples described in this
section. It also contains a raw CUDA version of the last RAJA CUDA example 
described here for comparison. 
