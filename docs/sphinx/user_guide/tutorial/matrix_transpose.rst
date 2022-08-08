.. ##
.. ## Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _matrixtranspose-label:

----------------------
Matrix Transpose
----------------------

Key RAJA features shown in this example are:

  * ``RAJA::kernel`` usage with multiple lambdas

The files ``RAJA/exercises/kernel-matrix-transpose_solution.cpp`` and
``RAJA/exercises/launch-matrix-transpose_solution.cpp`` contain the complete
solutions for the examples described in this section, including OpenMP, CUDA,
and HIP variants.

In this basic example, we compute the transpose of an input matrix
:math:`A` of size :math:`N_r \times N_c` and store the result in a second
matrix :math:`At` of size :math:`N_c \times N_r`.

We start with a non-RAJA C++ implementation. First we define our matrix dimensions.

.. literalinclude:: ../../../../exercises/kernel-matrix-transpose_solution.cpp
   :start-after: // _mattranspose_dims_start
   :end-before: // _mattranspose_dims_end
   :language: C++

Then, we wrap the matrix data pointers in ``RAJA::View`` objects to
simplify the multi-dimensional indexing:

.. literalinclude:: ../../../../exercises/kernel-matrix-transpose_solution.cpp
   :start-after: // _mattranspose_views_start
   :end-before: // _mattranspose_views_end
   :language: C++

Then, the non-RAJA C++ implementation looks like this:

.. literalinclude:: ../../../../exercises/kernel-matrix-transpose_solution.cpp
   :start-after: // _cstyle_mattranspose_start
   :end-before: // _cstyle_mattranspose_end
   :language: C++

^^^^^^^^^^^^^^^^^^^^^^^^^^^
RAJA::kernel Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For ``RAJA::kernel`` variants, we use ``RAJA::statement::For`` types
for the loops. The complete sequential RAJA variant is:

.. literalinclude:: ../../../../exercises/kernel-matrix-transpose_solution.cpp
   :start-after: // _raja_mattranspose_start
   :end-before: // _raja_mattranspose_end
   :language: C++

To execute the ``RAJA::kernel`` variant on the GPU we must redefine our execution
policy. The complete CUDA implementation is:

.. literalinclude:: ../../../../exercises/kernel-matrix-transpose_solution.cpp
   :start-after: // _raja_mattranspose_cuda_start
   :end-before: // _raja_mattranspose_cuda_end
   :language: C++

An interactive exercise for matrix-transpose can be found at
``RAJA/exercises/kernel-matrix-transpose.cpp``. 

^^^^^^^^^^^^^^^^^^^^^^^^^^^
RAJA::expt::launch Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For ``RAJA::expt::launch`` variants, we use ``RAJA::expt::loop`` methods to express
the hierachy of loops within the kernel execution space. For a sequential dispatch
we template the launch method using the ``RAJA::expt::seq_launch_t`` type and the loop methods
with ``RAJA::loop_exec``. The complete sequential RAJA variant is:

.. literalinclude:: ../../../../exercises/launch-matrix-transpose_solution.cpp
   :start-after: // _raja_mattranspose_start
   :end-before: // _raja_mattranspose_end
   :language: C++

To execute the ``RAJA::expt::launch`` variant on the GPU we must redefine our execution
policy. The complete CUDA implementation is:

.. literalinclude:: ../../../../exercises/launch-matrix-transpose_solution.cpp
   :start-after: // _raja_mattranspose_cuda_start
   :end-before: // _raja_mattranspose_cuda_end
   :language: C++

A notable difference between running on the GPU in contrast to the CPU using ``RAJA::expt::launch``
is the construction of the compute grid; the implicit construction of the compute grid is unique
to ``RAJA::kernel`` as it determines number of CUDA threads and blocks based on the provided
``RAJA::RangeSegment``.

A working exercise for matrix-transpose can be found at ``RAJA/exercises/launch-matrix-transpose.cpp``.
