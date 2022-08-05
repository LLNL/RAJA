.. ##
.. ## Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _view_layout-label:

-----------------------------------------------------------
RAJA Views and Layouts
-----------------------------------------------------------

This section contains an exercise to work through in the file
``RAJA/exercises/view-layout.cpp``.

Key RAJA features shown in this section are:

  * ``RAJA::View`` 
  * ``RAJA::Layout`` 
  * ``RAJA::OffsetLayout``

The file ``RAJA/exercises/view-layout_solution.cpp`` 
contains complete working code for the examples discussed in this section.

The examples in this section illustrate RAJA View and Layout concepts
and usage patterns. The goal is for you to gain an understanding of how
to use RAJA Views and Layouts to simplify and transform array data access
operations. None of the examples use RAJA kernel execution methods, such
as ``RAJA::forall``. The initent is to focus on RAJA View and Layout mechanics.

Consider a basic C-style implementation of a matrix-matrix multiplication
operation, where we use :math:`N \times N` matrices:

.. literalinclude:: ../../../../exercises/view-layout_solution.cpp
   :start-after: _cstyle_matmult_start
   :end-before: _cstyle_matmult_end
   :language: C++

Here, we use the array ``Cref`` to hold a reference solution matrix that
we use to compare with below. As is commonly done for efficiency in C and C++, 
we have allocated the data for the matrices as one-dimensional arrays. Thus, 
we need to manually compute the data pointer offsets for the row and column 
indices in the kernel.

To simplify the multi-dimensional indexing, we can use ``RAJA::View`` objects,
which we define as:

.. literalinclude:: ../../../../exercises/view-layout_solution.cpp
   :start-after: _matmult_views_start
   :end-before: _matmult_views_end
   :language: C++

Here we define three ``RAJA::View`` objects (``Aview``, ``Bview``, ``Cview``)
that *wrap* the array data pointers (``A``, ``B``, ``C``). We pass a data 
pointer as the first argument to each view constructor and then the extent of
each matrix dimension as the second and third arguments. The matrices are 
square and each extent is ``N``. Here, the template parameters to ``RAJA::View``
are the array data type, ``double``, and a ``RAJA::Layout``. Specifically::

  RAJA::Layout<2, int>

means that each View represents a two-dimensional default data layout, and that
we will use index values of type ``int`` to index into the arrays. 

Using the ``RAJA::View`` objects, we can access the data entries for the rows 
and columns using a more natural, less error-prone syntax:

.. literalinclude:: ../../../../exercises/view-layout_solution.cpp
   :start-after: _cstyle_matmult_views_start
   :end-before: _cstyle_matmult_views_end
   :language: C++

Default layouts use row-major data ordering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default data layout ordering in RAJA is *row-major*, which is the 
convention for multi-dimensional array indexing in C and C++. This means that 
the rightmost index will be stride-1, the index to the left of the rightmost 
index will have stride equal to the extent of the rightmost dimension, etc. 
This is described in more detail in :ref:`view-label`.

To illustrate the default data layout striding, we next show simple
one-, two-, and three-dimensional examples where the for-loop ordering 
for the different dimensions is such that all data access is stride-1. We 
begin by defining some dimensions, allocate and initialize arrays:

.. literalinclude:: ../../../../exercises/view-layout_solution.cpp
   :start-after: _default_views_init_start
   :end-before: _cstyle_matmult_views_end
   :language: C++

The version of the array initialization kernel using a one-dimensional 
``RAJA::View`` is:

.. literalinclude:: ../../../../exercises/view-layout_solution.cpp
   :start-after: _default_view1D_start
   :end-before: _default_view1D_end
   :language: C++

The version of the array initialization using a two-dimensional 
``RAJA::View`` is:

.. literalinclude:: ../../../../exercises/view-layout_solution.cpp
   :start-after: _default_view2D_start
   :end-before: _default_view2D_end
   :language: C++

The three-dimensional version is: 

.. literalinclude:: ../../../../exercises/view-layout_solution.cpp
   :start-after: _default_view3D_start
   :end-before: _default_view3D_end
   :language: C++

It's worth repeating that the data array access in all three variants using 
``RAJA::View`` objects is strice-1. Next, we will show how to permute 
the data striding order using permuted RAJA Layouts.

Permuted layouts change the data striding order
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

