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

.. _vertexsum-label:

--------------------------------------------------
Mesh Vertex Sum Example: Iteration Space Coloring
--------------------------------------------------

Key RAJA features shown in this example:

  * ``RAJA::forall`` loop iteration template method
  * ``RAJA::ListSegment`` iteration space construct
  * ``RAJA::IndexSet`` iteration space segment container and associated 
  * execution policies


The example a sum at each vertex on a logically-Cartesian 2D mesh. Each sum 
includes a contribution from each mesh element that share the vertex. In many 
"staggered mesh" applications, such operations are common and are usually
written in a way that prevents parallelization due to potential data races.
Specifically, multiple loop iterates over mesh elements may be writing to 
the same shared vertex memory location. The example illustrates how RAJA 
contructs can be used to enable one to extract parallelism (and potentially
improved performance) from such operations without fundamentally changing 
how the algorithm looks in source code.

The file ``RAJA/examples/ex6-vertexsum-coloring.cpp`` contains the complete 
working example code.
