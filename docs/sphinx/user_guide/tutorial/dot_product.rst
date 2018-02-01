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

.. _dotproduct-label:

-----------------------------------
Vector Dot Product (Sum Reduction)
-----------------------------------

Key RAJA features shown in this example:

  * ``RAJA::forall`` loop traversal template
  * ``RAJA::RangeSegment`` iteration space construct
  * RAJA execution policies
  * ``RAJA::ReduceSum`` sum reduction template
  * RAJA reduction policies


In the example, we compute a vector dot product, 'dot = (a,b)', where 
'a' and 'b' are two vectors length N and 'dot' is a scalar. A typical
C-style loop for the dot product is: 

.. literalinclude:: ../../../../examples/ex2-dot-product.cpp
                    :lines: 81-87

^^^^^^^^^^^^^^^^^^^^^
RAJA Variants
^^^^^^^^^^^^^^^^^^^^^

Different programming models support parallel reduction operations differently.
Some models, such as CUDA, do not provide support for reductions at all and 
so such operations must be explicitly coded by users. It can be challenging 
to generate a correct and high performance implementation. RAJA provides 
portable reduction types that make it easy to perform reduction operations
in loop kernels. The RAJA variants of the dot product computation show how 
to use the ``RAJA::ReduceSum`` sum reduction template type. RAJA provides
other reduction types and also allows multiple reduction operations to be
performed in a single kernel along with other computation. Please see 
:ref:`reductions-label` for an example that does this.

Each RAJA reduction type takes a `reduce policy` template argument. The type
of the reduction must match the type of the execution policy for the loop
in which it will be used. Here is the RAJA sequential variant of the dot 
product computation:

.. literalinclude:: ../../../../examples/ex2-dot-product.cpp
                    :lines: 95-102

Note that the sum reduction object is defined by specifying the sequential
reduction policy ``RAJA::seq_reduce``, which matches the loop execution policy,
and reduction value type (i.e., 'double'). An initial value of zero for the 
sum is also provided. To retrieve the reduction value after the loop is run,
we use the 'get' method.

The OpenMP multi-threaded variant of the loop is implemented similarly:

.. literalinclude:: ../../../../examples/ex2-dot-product.cpp
                    :lines: 112-119

Here, we use the ``RAJA::omp_reduce`` reduce policy to match the OpenMP
loop execution policy.

Finally, the RAJA CUDA variant:

.. literalinclude:: ../../../../examples/ex2-dot-product.cpp
                    :lines: 130-138

Here, the CUDA reduce policy ``RAJA::cuda_reduce`` is used to match the CUDA 
loop execution policy, including the number of threads per CUDA thread block.

It is worth noting how similar the code looks for each of these variants.
The loop body is identical for each and only the loop execution and policy
types change.

The file ``RAJA/examples/ex2-dot-product.cpp`` contains the complete 
working example code.
