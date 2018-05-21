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

.. _offset-label:

---------------------------------------------
Offsets for RAJA Views
---------------------------------------------

Key RAJA features shown in the following example:

  * ``RAJA::forall`` loop traversal template 
  * RAJA execution policies
  * ``RAJA::View`` multi-dimensional data access
  * ``RAJA:make_offset_layout`` layouts which offset data locations in an array

In this section, we present examples which apply a five-cell
stencil to the interior cells of a lattice of size :math:`(N + 2) \times (N + 2)`. 
A five-cell stencil accumulates the sum of a cell and its four cartesian neighbors
(North, South, East, and West). 
The interior :math:`N \times N` cells are assumed to be intialized to one. The result
of the stencil computation is stored in a second lattice of the same size. In the case of :math:`N=2` the
intial lattice takes the form:

  +---+---+---+---+
  | 0 | 0 | 0 | 0 |
  +---+---+---+---+
  | 0 | 1 | 1 | 0 |
  +---+---+---+---+
  | 0 | 1 | 1 | 0 |
  +---+---+---+---+
  | 0 | 0 | 0 | 0 |
  +---+---+---+---+

The second lattice may then be expected to be of the form:

  +---+---+---+---+
  | 0 | 0 | 0 | 0 |
  +---+---+---+---+
  | 0 | 3 | 3 | 0 |
  +---+---+---+---+
  | 0 | 3 | 3 | 0 |
  +---+---+---+---+
  | 0 | 0 | 0 | 0 |
  +---+---+---+---+


To avoid cells on the edge of the lattice we consider the following cell enumeration

  +----------+---------+---------+---------+
  | (-1, 2)  | (0, 2)  | (1, 2)  | (2, 2)  |
  +----------+---------+---------+---------+
  | (-1, 1)  | (0, 1)  | (1, 1)  | (2, 1)  |
  +----------+---------+---------+---------+
  | (-1, 0)  | (0, 0)  | (1, 0)  | (2, 0)  |
  +----------+---------+---------+---------+
  | (-1, -1) | (0, -1) | (1, -1) | (-2, 1) |
  +----------+---------+---------+---------+

and apply our stencil within the range of :math:`[0, 1] \times [0, 1]`.
To motivate the use of the ``RAJA::make_offset_layout`` and ``RAJA::Views`` we introduce here,
we define the following macros to access the matrix entries as would be done in a classic C-version.

.. literalinclude:: ../../../../examples/ex-offset.cpp
                    :lines: 91-92

With the macros in place, a typical C-style implemention may look like this:

.. literalinclude:: ../../../../examples/ex-offset.cpp
                    :lines: 141-148


^^^^^^^^^^^^^^^^^^^^
RAJA Kernel Variants
^^^^^^^^^^^^^^^^^^^^
In the RAJA variants of the stencil examples, we use two ``RAJA::Range Segment`` objects to 
define the row and column iteration spaces for the interior of the cell:

.. literalinclude:: ../../../../examples/ex-offset.cpp
                    :lines: 160-161

Offsets are generated for the ``RAJA::View`` by using a custom layout which effectively
shifts the start and end position of the arrays.

.. literalinclude:: ../../../../examples/ex-offset.cpp
                    :lines: 176-178

The first array in ``RAJA::make_offset_layout<DIM>`` corresponds to the starting index for the row and column
indices, while the second array corresponds to the inclusive ends of the arrays. The dimension of the ``view`` is specified by ``DIM``. 
The following ``RAJA`` example carries out the stencil operations using the ``RAJA::kernel`` framework, we refer the reader 
to the example :ref:`matrixmultiply-label` for an introduction to the ``RAJA::kernel`` framework.

.. literalinclude:: ../../../../examples/ex-offset.cpp
                    :lines: 191-197

The file ``RAJA/examples/ex-offset.cpp`` contains the complete working example code. 
