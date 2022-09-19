.. ##
.. ## Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _matrixmultiply-label:

------------------------------------
Matrix Multiplication (Nested Loops)
------------------------------------

Key RAJA features shown in the following examples:

  * ``RAJA::kernel`` template for nested-loop execution
  * RAJA kernel execution policies
  * ``RAJA::View`` multi-dimensional data access
  * Basic RAJA nested-loop interchange 
  * Specifying lambda arguments through statements

In this example, we present different ways to perform multiplication of two 
square matrices 'A' and 'B' of dimension N x N and store the result in matrix 
'C'. To motivate the use of the ``RAJA::View`` abstraction that we use, 
we define the following macros to access the matrix entries in the 
C-version:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_macros_start
   :end-before: _matmult_macros_end
   :language: C++

Then, a typical C-style sequential matrix multiplication operation looks like
this:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_cstyle_start
   :end-before: _matmult_cstyle_end
   :language: C++

For the RAJA variants of the matrix multiple operation presented below,
we use ``RAJA::View`` objects, which allow us to access matrix
entries in a multi-dimensional manner similar to the C-style version that
uses macros. We create a two-dimensional N x N 'view'
for each of the three matrices:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_views_start
   :end-before: _matmult_views_end
   :language: C++

We show the most basic RAJA view usage here -- to simplify multi-dimensional
array indexing. RAJA views can be used to abstract a variety of different 
data layouts and access patterns, including stride permutations, offsets, etc. 
For more information about RAJA views, see :ref:`view-label`.

We also use the following ``RAJA::RangeSegment`` objects to define the matrix 
row and column and dot product iteration spaces:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_ranges_start
   :end-before: _matmult_ranges_end
   :language: C++

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Should I Use RAJA::forall For Nested Loops?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We begin by walking through some RAJA variants of the matrix multiplication 
operation that show RAJA usage that **we do not recommend**, but which helps
to motivate the ``RAJA::kernel`` interface. We noted some rationale behind 
this preference in :ref:`loop_elements-kernel-label`. Here, we discuss this
in more detail.

Starting with the C-style kernel above, we first convert the outermost 
'row' loop to a ``RAJA::forall`` method call with a sequential execution policy:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_outerforall_start
   :end-before: _matmult_outerforall_end
   :language: C++

Here, the lambda expression for the loop body contains the inner 
'col' and 'k' loops.

Note that changing the RAJA execution policy to an OpenMP or CUDA policy
enables the outer 'row' loop to run in parallel. When this is done, 
each thread executes the lambda expression body, which contains the 'col' 
and 'k' loops. Although this enables some parallelism, there is still more 
available. In a bit, we will how the ``RAJA::kernel`` interface helps us to
expose all available parallelism.

Next, we nest a ``RAJA::forall`` method call for the 'column' loop inside the 
outer lambda expression:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_nestedforall_start
   :end-before: _matmult_nestedforall_end
   :language: C++

Here, the innermost lambda expression contains the row-column dot product
initialization, the inner 'k' loop for the dot product, and the operation
that assigns the dot product to the proper location in the result matrix. 

Note that we can replace either RAJA execution policy with an OpenMP 
execution policy to parallelize either the 'row' or 'col' loop. For example, 
we can use an OpenMP execution policy on the outer 'row' loop and the result 
will be the same as using an OpenMP execution policy in the earlier case that 
used a ``RAJA::forall`` statement for the outer loop.

We do not recommend using a parallel execution policy for both loops in 
this type of kernel as the results may not be what is expected and RAJA 
provides better mechanisms for parallelizing nested loops. Also, changing 
the outer loop policy to a CUDA policy will not compile. This is by design 
in RAJA since nesting forall statements inside lambdas in this way has limited 
utility, is inflexible, and can hinder performance when compared to 
``RAJA::kernel`` constructs, which we describe next. 

.. _matmultkernel-label:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Basic RAJA::kernel Variants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next, we show how to cast the matrix-multiplication operation 
using the ``RAJA::kernel`` interface, which was introduced in
:ref:`loop_elements-kernel-label`. We first present a complete example, and
then describe its key elements, noting important differences between
``RAJA::kernel`` and ``RAJA::forall`` loop execution interfaces. 

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_basickernel_start
   :end-before: _matmult_basickernel_end
   :language: C++

Here, ``RAJA::kernel`` expresses the outer 'row' and 'col' loops; 
the inner 'k' loop is included in the lambda expression for the loop body. 
Note that the ``RAJA::kernel`` template takes two arguments. Similar to 
``RAJA::forall``, the first argument describes the iteration space and the 
second argument is the lambda loop body. Unlike ``RAJA::forall``, the 
iteration space for ``RAJA::kernel`` is defined as a *tuple* of ranges 
(created via the ``RAJA::make_tuple`` method), one for the 'col' loop and 
one for the 'row' loop. Also, the lambda expression takes an iteration index 
argument for entry in the iteration space tuple. 

.. note :: The number and order of lambda arguments must match the number and
           order of the elements in the tuple for this to be correct.

Another important difference between ``RAJA::forall`` and ``RAJA::kernel`` 
involves the execution policy template parameter. The execution policy defined 
by the ``RAJA::KernelPolicy`` type used here specifies a policy for each level 
in the loop nest via nested ``RAJA::statement::For`` types. Here, the row and 
column loops will both execute sequentially. The integer that appears as the 
first template parameter to each 'For' statement corresponds to the position of 
a range in the iteration space tuple and also to the associated iteration 
index argument to the lambda. Here, '0' is the 'col' range and '1' is the 
'row' range because that is the order those ranges appear in the tuple. The 
innermost type ``RAJA::statement::Lambda<0>`` indicates that the first lambda
expression (the only one in this case!) argument passed to the 
``RAJA::kernel`` method will be invoked inside the nested loops.

The integer arguments to the ``RAJA::statement::For`` types are needed to 
enable a variety of kernel execution patterns and transformations. Since the 
kernel policy is a single unified construct, it can be used to parallelize 
the nested loop iterations together, which we will show later. Also, the 
levels in the loop nest can be permuted by reordering the policy arguments; 
this is analogous to how one would reorder C-style nested loops; i.e., 
reorder for-statements for each loop nest level. These execution patterns 
and transformations can be achieved by changing only the policy and leaving the 
loop kernel code as is.

If we want to execute the row loop using OpenMP multithreaded parallelism 
and keep the column loop sequential, the policy we would use is:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_ompkernel_start
   :end-before: _matmult_ompkernel_end
   :language: C++

To swap the loop nest ordering and keep the same execution policy on each loop,
we would use the following policy, which swaps the ``RAJA::statement::For`` 
types. The inner loop is now the 'row' loop and is run in parallel; 
the outer loop is now the 'col' loop and is still sequential:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_ompkernel_swap_start
   :end-before: _matmult_ompkernel_swap_end
   :language: C++

.. note:: It is important to emphasize that these kernel transformations, 
          and others, can be done by switching the ``RAJA::KernelPolicy`` 
          type with no changes to the loop kernel code.

In :ref:`kernelnestedreorder-label`, we provide a more detailed discussion of the
mechanics of loop nest reordering. Next, we show other variations of the 
matrix multiplication kernel that illustrate other ``RAJA::kernel`` features. 

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
More Complex RAJA::kernel Variants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The matrix multiplication kernel variations described in this section use
execution policies to express the outer row and col loops as well as the 
inner dot product loop using the RAJA kernel interface. They illustrate more 
complex policy examples and show additional RAJA kernel features.

The first example uses sequential execution for all loops:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_3lambdakernel_seq_start
   :end-before: _matmult_3lambdakernel_seq_end
   :language: C++

Note that we use a ``RAJA::kernel_param`` method to execute the kernel. It is
similar to ``RAJA::kernel`` except that it accepts a tuple as the second
argument (between the iteration space tuple and the lambda expressions). The
tuple is a set of *parameters* that can be used in the kernel to pass data
into lambda expressions. Here, the parameter tuple holds a single scalar 
variable for the dot product. 

The remaining arguments include a sequence of lambda expressions representing 
different parts of the inner loop body. We use three lambda expressions that: 
initialize the dot product variable (lambda 0), define the 'k' inner loop 
row-col dot product operation (lambda 1), and store the computed row-col dot 
product in the proper location in the result matrix (lambda 2). Note that
all lambdas take the same arguments in the same order, which is required for 
the kernel to be well-formed. In addition to the loop index variables, we 
pass the scalar dot product variable into each lambda. This enables the same 
variables to be used in all three lambda expressions. However, observe that
not all lambda expressions use all three index variables. They are declared, 
but left unnamed to prevent compiler warnings. 

Alternatively, the lambda statements in the execution policy may be used
to specify which arguments each lambda takes and in which order. For example:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_3lambdakernel_args_seq_start
   :end-before: _matmult_3lambdakernel_args_seq_end
   :language: C++

By using ``RAJA::statement::Lambda`` parameters in this way, the code 
potentially indicates more clearly which areguments are used. Of course, this 
makes the execution policy more verbose, but that is typically hidden away 
in a header file. Statements such as ``RAJA::Segs``, and 
``RAJA::Params`` identify the positions of the segments and params 
in the tuples to be used as arguments to the lambda expressions.

As we noted earlier, the execution policy type passed to the 
``RAJA::kernel_param`` method as a template parameter describes how the 
statements and lambda expressions are assembled to form the complete kernel. 
To illustrate this, we describe various policies that enable the kernel to 
run in different ways. In each case, the ``RAJA::kernel_param`` method call, 
including its arguments is the same. The curious reader will inspect the 
example code in the file listed below to see that this is indeed the case.
In the interest of simplicity, the remaining matrix multiplication examples
do not use ``RAJA::statement::Lambda`` parameters to specify arguments to
the lambda expressions.

Next, we show how to collapse nested loops in an OpenMP parallel region
using a ``RAJA::statement::Collapse`` type in the execution policy. This
allows one to parallelize multiple levels in a loop nest using OpenMP 
directives, for instance. The following policy will collapse the two outer 
loops into one OpenMP parallel region:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_3lambdakernel_ompcollapse_start
   :end-before: _matmult_3lambdakernel_ompcollapse_end
   :language: C++

The ``RAJA::ArgList`` type indicates which loops in the nest are to be 
collapsed and their nesting order within the collapse region. The integers
passed to ``ArgList`` are indices of entries in the tuple of iteration spaces 
and indicate inner to outer loop levels when read from right to left (i.e., 
here '1, 0' indicates the column loop is the inner loop and the row loop is 
the outer). For this transformation there are no ``statement::For`` types
and policies for the individual loop levels inside the OpenMP collapse region. 

Lastly, we show how to use ``RAJA::statement::CudaKernel`` and 
``RAJA::statement::HipKernel`` types to generate GPU kernels launched with
a particular thread-block decomposition.  We reiterate that although the 
policies are different, the kernels themselves are identical to the sequential 
and OpenMP variants above.

Here is a policy that will distribute the row indices across CUDA thread 
blocks and column indices across threads in the x dimension of each block:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_3lambdakernel_cuda_start
   :end-before: _matmult_3lambdakernel_cuda_end
   :language: C++

This is equivalent to defining a CUDA kernel with the lambda body inside
it and defining row and column indices as::

  int row = blockIdx.x;
  int col = threadIdx.x;  

and launching the kernel with appropriate CUDA grid and thread-block dimensions.

The HIP execution policy is similar:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_3lambdakernel_hip_start
   :end-before: _matmult_3lambdakernel_hip_end
   :language: C++


The following policy will tile row and col indices across two-dimensional
CUDA thread blocks with 'x' and 'y' dimensions defined by a 'CUDA_BLOCK_SIZE'
parameter that can be set at compile time. Within each tile, the kernel 
iterates are executed by CUDA threads.

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_3lambdakernel_cudatiled_start
   :end-before: _matmult_3lambdakernel_cudatiled_end
   :language: C++

Note that the tiling mechanism requires a ``RAJA::statement::Tile`` type, 
with a tile size and a tiling execution policy, plus a ``RAJA::statement::For``
type with an execution execution policy for each tile dimension.

The analogous HIP policy is:

.. literalinclude:: ../../../../examples/tut_matrix-multiply.cpp
   :start-after: _matmult_3lambdakernel_hiptiled_start
   :end-before: _matmult_3lambdakernel_hiptiled_end
   :language: C++

In :ref:`tiledmatrixtranspose-label` and :ref:`matrixtransposelocalarray-label`,
we will discuss loop tiling in more detail including how it can be used to 
improve performance of certain algorithms.

The file ``RAJA/examples/tut_matrix-multiply.cpp`` contains the complete 
working code for all examples described in this section, plus others that
show a variety of ``RAJA::kernel`` execution policy types. It also contains 
a raw CUDA version of the kernel for comparison.
