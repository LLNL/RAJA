.. ##
.. ## Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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

  * ``RAJA::forall`` loop execution template method
  * ``RAJA::ListSegment`` iteration space construct
  * ``RAJA::IndexSet`` iteration space segment container and associated execution policies


The example computes a sum at each vertex on a logically-Cartesian 2D mesh
as shown in the figure.

.. figure:: ../figures/vertexsum.jpg

   A portion of the area of each mesh element is summed to the vertices surrounding the element.

Each sum is an average of the area of the mesh elements that share the vertex. 
In many "staggered mesh" applications, such an operation is common and is 
often written in a way that presents the algorithm clearly but prevents 
parallelization due to potential data races. That is, multiple loop iterates 
over mesh elements may attempt to write to the same shared vertex memory 
location at the same time. The example shows how RAJA constructs can be 
used to enable one to express such an algorithm in parallel and have it
run correctly without fundamentally changing how it looks in source code.

After defining the number of elements in the mesh, necessary array offsets
and an array that indicates the mapping between an element and its four 
surrounding vertices, a C-style version of the vertex sum calculation is:

.. literalinclude:: ../../../../examples/tut_vertexsum-coloring.cpp
                    :lines: 122-131

^^^^^^^^^^^^^^^^^^^^^^^
RAJA Sequential Variant
^^^^^^^^^^^^^^^^^^^^^^^

A nested loop RAJA variant of this kernel is:

.. literalinclude:: ../../../../examples/tut_vertexsum-coloring.cpp
                    :lines: 143-161

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
(or `colors`) indicated by the numbers in the figure below, which represents
a portion of our logically-Cartesian 2D mesh.

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

.. literalinclude:: ../../../../examples/tut_vertexsum-coloring.cpp
                    :lines: 198-220

Then, we create a RAJA index set with four list segments, one for each color,
using the vectors:

.. literalinclude:: ../../../../examples/tut_vertexsum-coloring.cpp
                    :lines: 228-235

Now, we can use an index set execution policy that iterates over the 
segments sequentially and executes each segment in parallel using OpenMP
multi-threading (and ``RAJA::forall``):

.. literalinclude:: ../../../../examples/tut_vertexsum-coloring.cpp
                    :lines: 278-287

We no longer need to use the offset variable to compute the 
element index in terms of 'i' and 'j' since the loop is no longer nested
and the element indices are directly encoded in the list segments.

For completeness, here is the RAJA variant where we iterate over the 
segments sequentially, and execute each segment in parallel via a CUDA
kernel launch on a GPU:

.. literalinclude:: ../../../../examples/tut_vertexsum-coloring.cpp
                    :lines: 306-315

Here, we have marked the lambda loop body with the 'RAJA_DEVICE' macro
and specified the number of threads in a CUDA thread block in the segment
execution policy.

The file ``RAJA/examples/tut_vertexsum-coloring.cpp`` contains the complete 
working example code.
