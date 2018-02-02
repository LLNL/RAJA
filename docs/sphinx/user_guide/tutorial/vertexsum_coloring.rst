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

.. _vertexsum-label:

--------------------------------------------------
Mesh Vertex Sum Example: Iteration Space Coloring
--------------------------------------------------

Key RAJA features shown in this example:

  * ``RAJA::forall`` loop iteration template method
  * ``RAJA::ListSegment`` iteration space construct
  * ``RAJA::IndexSet`` iteration space segment container and associated execution policies


The example computed a sum at each vertex on a logically-Cartesian 2D mesh. 
Each sum is an average of the area of the mesh elements that share the vertex. 
In many "staggered mesh" applications, such an operation is common and is 
often written in a way that presents the algorithm clearly but prevents 
parallelization due to potential data races. Specifically, multiple loop 
iterates over mesh elements may be writing to the same shared vertex memory 
location. The example illustrates how RAJA constructs can be used to enable 
one to extract parallelism (and potentially improved performance) from such 
an algorithm without fundamentally changing how it looks in source code.

After defining the number of elements in the mesh, necessary array offsets
and an array that indicates the mapping between an element and its four 
surrounding vertices, the C-style version of the vertex sum calculation is:

.. literalinclude:: ../../../../examples/ex6-vertexsum-coloring.cpp
                    :lines: 122-131

^^^^^^^^^^^^^^^^^^^^^^^
RAJA Sequential Variant
^^^^^^^^^^^^^^^^^^^^^^^

The nested loop RAJA variant of this kernel is:

.. literalinclude:: ../../../../examples/ex6-vertexsum-coloring.cpp
                    :lines: 143-157

Note that this version cannot be guaranteed to run correctly in parallel
by simply changing the loop execution policies as we have done in other
examples. We would like to use RAJA to enable parallel execution and without
changing the way the kernel looks in source code. By applying a RAJA index
set and suitably-defined list segments, we can accomplish this.

^^^^^^^^^^^^^^^^^^^^^^^
RAJA Parallel Variants
^^^^^^^^^^^^^^^^^^^^^^^

To enable the kernel to run safely in parallel, by eliminating the race 
conditions, we partition the element iteration space into four subsets
(or `colors`) indicated by the numbers in the figure below.

  +---+---+---+---+
  | 2 | 3 | 2 | 3 |
  +---+---+---+---+
  | 0 | 1 | 0 | 1 |
  +---+---+---+---+
  | 2 | 3 | 2 | 3 |
  +---+---+---+---+
  | 0 | 1 | 0 | 1 |
  +---+---+---+---+

Note that none of the elements with the same number share a common vertex. 
Thus, we can iterate over all elements with the same number (i.e., color) 
in parallel.

First, we define four vectors to gather the mesh element indices for each
color:

.. literalinclude:: ../../../../examples/ex6-vertexsum-coloring.cpp
                    :lines: 194-216

Then, we create a RAJA index set with four list segments, one for each color:

.. literalinclude:: ../../../../examples/ex6-vertexsum-coloring.cpp
                    :lines: 224-231

Now, we can define an index set execution policy that iterates over the 
segments sequentially and traverses each segment in parallel using OpenMP
multi-threading:

.. literalinclude:: ../../../../examples/ex6-vertexsum-coloring.cpp
                    :lines: 274-283

Note that we no longer need to use the offset variable to compute the 
element index in terms of 'i' and 'j' since the loop is no longer nested
and the element indices are directly encoded in the list segments.

For completeness, here is the RAJA variant where we iterate over the 
segments sequentially, and execute each segment in parallel via a CUDA
kernel launch on a GPU:

.. literalinclude:: ../../../../examples/ex6-vertexsum-coloring.cpp
                    :lines: 302-311

Note that we've marked the lambda loop body with the 'RAJA_DEVICE' macro
and specified the number of threads in a CUDA thread block in the segment
execution policy.

The file ``RAJA/examples/ex6-vertexsum-coloring.cpp`` contains the complete 
working example code.
