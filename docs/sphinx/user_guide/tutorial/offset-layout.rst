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

  * ``RAJA::Kernel`` loop traversal template
  *  RAJA execution policies
  * ``RAJA::View`` multi-dimensional data access
  * ``RAJA:make_offset_layout`` Method which returns a layout with offset enumeration for each index

This example applies a five-cell stencil to the
interior cells of a lattice and stores the
resulting sums in a second lattice of equal size.
The five-cell stencil accumulates values of a cell
and its four neighbors. Assuming the cells of a
lattice may be accessed in a row/col fashion through a parenthesis operator,
the stencil may be expressed as the following sum::

  output_lattice(row, col)
    = input_lattice(row, col)
    + input_lattice(row - 1, col) + input_lattice(row + 1, col)
    + input_lattice(row, col - 1) + input_lattice(row, col + 1)

A lattice is assumed to have :math:`N_r \times N_c` interior nodes with unit values and a padded edge
of zeros for a total dimension of :math:`(N_r + 2) \times (N_c + 2)`. In the case of
:math:`N_r = N_c = 3`, the input lattice generated takes the form:

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

After applying the stencil, the output lattice is expected to
take the form:

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

In this example, the ``RAJA::make_offset_layout``
method and ``RAJA::View`` object are used to simplify applying
the stencil to interior cells. The make_offset_layout method
enables developers to offset the enumeration of values in an array.
Here we choose to enumerate the lattice in the following manner:

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

Notably :math:`[0, N_r) \times [0, N_c)` corresponds to the index
range we wish to apply the stencil to, and :math:`[-1,N_r] \times [-1, N_c]`
are the range of coordinates of the lattice.

^^^^^^^^^^^^^^^^^^
RAJA Offset Layout
^^^^^^^^^^^^^^^^^^
As a first step, the code below uses the make_offset_layout method to construct a layout
to create a new enumeration for the two-dimensional array which represents the lattice.
In particular, the row index is chosen to be within the inclusive range of :math:`[-1, N_r]`,
and the column index is chosen to be within the inclusive range of :math:`[-1, N_c]`:

.. literalinclude:: ../../../../examples/ex-offset.cpp
                    :lines: 199-202

The arguments of the layout method are standard C++ arrays with the coordinates of
the bottom left corner of the lattice and the coordinates of the top right
corner of the lattice. The example uses double braces as it enables proper initiation
of the object and subobjects. We refer the reader to the :ref:`view-label` section
for a complete description of ``RAJA::View`` and ``RAJA::Layout`` objects.

^^^^^^^^^^^^^^^^^^^^
RAJA Kernel Variants
^^^^^^^^^^^^^^^^^^^^
For the RAJA variants of the stencil examples, two ``RAJA::Range Segment`` objects
are used to define the row and column iteration spaces for the interior of the cell:

.. literalinclude:: ../../../../examples/ex-offset.cpp
                    :lines: 182-183

As the stencil operation is data parallel, any ``RAJA::Kernel`` policy may be expected to
work with the loop body. The example below illustrates the computation within
the RAJA programming model. We refer the reader to :ref:`loop_elements-label`
and :ref:`matmultkernel-label` for an introduction and examples for the kernel framework.

.. literalinclude:: ../../../../examples/ex-offset.cpp
                    :lines: 209-227

The file ``RAJA/examples/ex-offset.cpp`` contains the complete working example code.
