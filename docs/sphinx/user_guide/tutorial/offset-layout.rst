.. ##
.. ## Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _offset-label:

---------------------------------------------
Stencil Computations (View Offsets)
---------------------------------------------

Key RAJA features shown in the following example:

  * ``RAJA::Kernel`` loop execution template
  *  RAJA kernel execution policies
  * ``RAJA::View`` multi-dimensional data access
  * ``RAJA:make_offset_layout`` method to apply index offsets

This example applies a five-cell stencil sum to the interior cells of a 
two-dimensional square lattice and stores the resulting sums in a second 
lattice of equal size. The five-cell stencil accumulates values from each
interior cell and its four neighbors. We use ``RAJA::View`` and 
``RAJA::Layout`` constructs to simplify the multi-dimensional indexing so 
that we can write the stencil operation as follows::

  output(row, col) = input(row, col) + 
                     input(row - 1, col) + input(row + 1, col) + 
                     input(row, col - 1) + input(row, col + 1)

A lattice is assumed to have :math:`N_r \times N_c` interior cells with unit 
values surrounded by a halo of cells containing zero values for a total 
dimension of :math:`(N_r + 2) \times (N_c + 2)`. For example, when
:math:`N_r = N_c = 3`, the input lattice and values are:

  +---+---+---+---+---+
  | 0 | 0 | 0 | 0 | 0 |
  +---+---+---+---+---+
  | 0 | 1 | 1 | 1 | 0 |
  +---+---+---+---+---+
  | 0 | 1 | 1 | 1 | 0 |
  +---+---+---+---+---+
  | 0 | 1 | 1 | 1 | 0 |
  +---+---+---+---+---+
  | 0 | 0 | 0 | 0 | 0 |
  +---+---+---+---+---+

After applying the stencil, the output lattice and values are:

  +---+---+---+---+---+
  | 0 | 0 | 0 | 0 | 0 |
  +---+---+---+---+---+
  | 0 | 3 | 4 | 3 | 0 |
  +---+---+---+---+---+
  | 0 | 4 | 5 | 4 | 0 |
  +---+---+---+---+---+
  | 0 | 3 | 4 | 3 | 0 |
  +---+---+---+---+---+
  | 0 | 0 | 0 | 0 | 0 |
  +---+---+---+---+---+

For this :math:`(N_r + 2) \times (N_c + 2)` lattice case, here is our 
(row, col) indexing scheme.

  +----------+---------+---------+---------+---------+
  | (-1, 3)  | (0, 3)  | (1, 3)  | (2, 3)  | (3, 3)  |
  +----------+---------+---------+---------+---------+
  | (-1, 2)  | (0, 2)  | (1, 2)  | (2, 2)  | (3, 2)  |
  +----------+---------+---------+---------+---------+
  | (-1, 1)  | (0, 1)  | (1, 1)  | (2, 1)  | (3, 1)  |
  +----------+---------+---------+---------+---------+
  | (-1, 0)  | (0, 0)  | (1, 0)  | (2, 0)  | (3, 0)  |
  +----------+---------+---------+---------+---------+
  | (-1, -1) | (0, -1) | (1, -1) | (2, -1) | (3, -1) |
  +----------+---------+---------+---------+---------+

Notably :math:`[0, N_r) \times [0, N_c)` corresponds to the interior index
range over which we apply the stencil, and :math:`[-1,N_r+1) \times [-1, N_c+1)`
is the full lattice index range.

^^^^^^^^^^^^^^^^^^^
RAJA Offset Layouts
^^^^^^^^^^^^^^^^^^^

We use the ``RAJA::make_offset_layout`` method to construct a 
``RAJA::OffsetLayout`` object that defines our two-dimensional indexing scheme.
Then, we create two ``RAJA::View`` objects for each of the input and output
lattice arrays.

.. literalinclude:: ../../../../examples/tut_offset-layout.cpp
   :start-after: _offsetlayout_views_start
   :end-before: _offsetlayout_views_end
   :language: C++

Here, the row index range is :math:`[-1, N_r+1)`, and the column index 
range is :math:`[-1, N_c+1)`. The first argument to each call to the 
``RAJA::View`` constructor is a pointer to an array that holds the data for 
the view; we assume the arrays are properly allocated before these calls.

The offset layout mechanics of RAJA allow us to write loops over
data arrays using non-zero based indexing and without having to manually 
compute the proper offsets into the arrays. For more details on the 
``RAJA::View`` and ``RAJA::Layout`` concepts we use in this example, please 
refer to :ref:`view-label`.

^^^^^^^^^^^^^^^^^^^^^^^^^^
RAJA Kernel Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^

For the RAJA implementations of the example computation, we use two 
``RAJA::RangeSegment`` objects to define the row and column iteration 
spaces for the interior cells:

.. literalinclude:: ../../../../examples/tut_offset-layout.cpp
   :start-after: _offsetlayout_ranges_start
   :end-before: _offsetlayout_ranges_end
   :language: C++

Here, is an implementation using ``RAJA::kernel`` multi-dimensional loop
execution with a sequential execution policy.

.. literalinclude:: ../../../../examples/tut_offset-layout.cpp
   :start-after: _offsetlayout_rajaseq_start
   :end-before: _offsetlayout_rajaseq_end
   :language: C++

Since the stencil operation is data parallel, any parallel execution policy 
may be used. The file ``RAJA/examples/tut_offset-layout.cpp`` contains a 
complete working example code with various parallel implementations. For more 
information about using the ``RAJA::kernel`` interface, please see 
:ref:`loop_elements-kernel-label`. 
