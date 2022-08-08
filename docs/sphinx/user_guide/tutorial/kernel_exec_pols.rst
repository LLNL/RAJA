.. ##
.. ## Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _kernelexecpols-label:

-----------------------------------------------------------
``RAJA::kernel`` Execution Policies
-----------------------------------------------------------

This section contains an exercise file ``RAJA/exercises/kernelintro-execpols.cpp``
for you to work through if you wish to get some practice with RAJA. The
file ``RAJA/exercises/kernelintro-execpols_solution.cpp`` contains
complete working code for the examples discussed in this section. You can use
the solution file to check your work and for guidance if you get stuck.

Key RAJA features shown in this section are:

  * ``RAJA::kernel`` loop iteration templates and execution policies 

The examples in this section illustrate various execution policies for
``RAJA::kernel``. The goal is for you to gain an understanding of how
execution policies are constructed and used to perform various nested
loop execution patterns. All examples use the same simple kernel, which
is a three-level loop nest to initialize the entries in a three-dimensional
tensor.

.. note:: The C++ lambda expression representing the kernel inner loop body
          is identical for all RAJA variants of the kernel, whether we are
          executing the kernel on a CPU sequentially or in parallel with 
          OpenMP, or in parallel on a GPU (CUDA or HIP).

We begin by defining some constants used throughout the examples and allocating
arrays to represent the tensor data:

.. literalinclude:: ../../../../exercises/kernelintro-execpols_solution.cpp
   :start-after: _init_define_start
   :end-before: _init_define_end
   :language: C++

Note that we use the 'memory manager' routines contained in the exercise
directory to simplify the allocation process. In particular, CUDA unified 
memory is used when CUDA is enabled to simplify accessing the data on the
host or device.

Next, we execute a C-style nested for-loop version of the kernel to initialize
the entries in the 'reference' array that we will use to compare the results
of other variants for correctness:

.. literalinclude:: ../../../../exercises/kernelintro-execpols_solution.cpp
   :start-after: _cstyle_tensorinit_seq_start
   :end-before: _cstyle_tensorinit_seq_end
   :language: C++

Note that we manually compute the pointer offsets for the (i,j,k) indices. 
To simplify the remaining kernel variants we introduce a ``RAJA::View``
object, which wraps the tensor data pointer and simplifies the multi-dimensional
indexing:

.. literalinclude:: ../../../../exercises/kernelintro-execpols_solution.cpp
   :start-after: _3D_raja_view_start
   :end-before: _3D_raja_view_end
   :language: C++

Here ``aView`` is a three-dimensional View with extent ``N`` in each
coordinate based on a three-dimensional ``RAJA::Layout`` object using
indices of type ``int``. Please see :ref:`view-label` for more information 
about the View and Layout types that RAJA provides for various indexing
patterns and data layouts.

Using the View, the C-style kernel looks like:

.. literalinclude:: ../../../../exercises/kernelintro-execpols_solution.cpp
   :start-after: _cstyle_tensorinit_view_seq_start
   :end-before: _cstyle_tensorinit_view_seq_end
   :language: C++

Notice how accessing each (i,j,k) entry in the tensor is more natural using
the View.

The corresponding RAJA sequential version using ``RAJA::kernel`` is:

.. literalinclude:: ../../../../exercises/kernelintro-execpols_solution.cpp
   :start-after: _raja_tensorinit_seq_start
   :end-before: _raja_tensorinit_seq_end
   :language: C++

This should be familiar to the reader who has read through the preceding
:ref:`kernelnestedreorder-label` section of this tutorial.

Suppose we wanted to parallelize the outer 'k' loop using OpenMP multithreading.
A C-style version of this is:

.. literalinclude:: ../../../../exercises/kernelintro-execpols_solution.cpp
   :start-after: _cstyle_tensorinit_omp_outer_start
   :end-before: _cstyle_tensorinit_omp_outer_end
   :language: C++

where we have placed the OpenMP directive ``#pragma omp parallel for`` before
the outer loop of the kernel.

To parallelize all iterations in the entire loop nest, we can apply the OpenMP
``collapse(3)`` clause to map the iterations for all loop levels to OpenMP 
threads:

.. literalinclude:: ../../../../exercises/kernelintro-execpols_solution.cpp
   :start-after: _cstyle_tensorinit_omp_collapse_start
   :end-before: _cstyle_tensorinit_omp_collapse_end
   :language: C++

The corresponding RAJA versions of these two C-style OpenMP variants are:

.. literalinclude:: ../../../../exercises/kernelintro-execpols_solution.cpp
   :start-after: _raja_tensorinit_omp_outer_start
   :end-before: _raja_tensorinit_omp_outer_end
   :language: C++

and 

.. literalinclude:: ../../../../exercises/kernelintro-execpols_solution.cpp
   :start-after: _raja_tensorinit_omp_collapse_start
   :end-before: _raja_tensorinit_omp_collapse_end
   :language: C++

The first of these, in which we parallelize the outer 'k' loop simply replaces
the ``RAJA::loop_exec`` loop execution policy with the 
``RAJA::omp_parallel_for_exec`` policy, which applies the same OpenMP
directive to the outer loop we described for the C-style variant.

The RAJA OpenMP collapse variant introduces the ``RAJA::statement::Collapse``
statement type. We use the ``RAJA::omp_parallel_collapse_exec`` execution
policy type and indicate that we want to collapse all three loop levels
in the second template argument ``RAJA::ArgList<2, 1, 0>``. The integer
values in the list indicate the order of the loops in the collapse operation:
'k' (2) outer, 'j' (1) middle, and 'i' (0) inner. The integers represent
the order of the lambda arguments and the order of the range segments in the
iteration space tuple.

The first RAJA-based kernel for parallel GPU execution using the RAJA CUDA
back-end we introduce is:

.. literalinclude:: ../../../../exercises/kernelintro-execpols_solution.cpp
   :start-after: _raja_tensorinit_cuda_start
   :end-before: _raja_tensorinit_cuda_end
   :language: C++

Here, we use the ``RAJA::statement::CudaKernel`` statement type to
indicate that we want a CUDA kernel to be launched. The 'k', 'j', 'i'
iteration variables are mapped to CUDA threads using the CUDA execution
policy types ``RAJA::cuda_thread_z_loop``, ``RAJA::cuda_thread_y_loop``,
and ``RAJA::cuda_thread_x_loop``, respectively. Thus, we use a 
a three-dimensional CUDA thread-block to map the loop iterations to CUDA
threads. The ``_loop`` part of each execution policy name indicates that
the indexing in the associated portion of the mapping will use a block-stride
loop. This is useful to guarantee that the policy will work for any 
tensor regardless of size in each coordinate dimension.

.. note:: A GPU device kernel body expressed as a C++ lambda expression 
          requires the ``__device__`` decoration. This can be inserted
          directly, as shown here, or using the ``RAJA_DEVICE`` macro.
          Using the RAJA macro provides flexibility and portability since
          it turns off the device decoration if the code is compiled with
          CUDA (or HIP) disabled.

To execute the kernel with a prescribed mapping of iterations to a
thread-block using RAJA, we could do the following:

.. literalinclude:: ../../../../exercises/kernelintro-execpols_solution.cpp
   :start-after: _raja_tensorinit_cuda_tiled_direct_start
   :end-before: _raja_tensorinit_cuda_tiled_direct_end
   :language: C++

where we have defined the CUDA thread-block dimensions as:

.. literalinclude:: ../../../../exercises/kernelintro-execpols_solution.cpp
   :start-after: _cuda_blockdim_start
   :end-before: _cuda_blockdim_end
   :language: C++

The ``RAJA::statement::CudaKernelFixed`` statement indicates that we want to 
use a fixed thread-block size of 256 with dimensions (32, 8, 1). To ensure that
we are mapping the kernel iterations properly in chunks of 256 threads to 
each thread-block, we use RAJA loop tiling statements in which we specify the
tile size for each dimension/loop index. For example, the statement
``RAJA::statement::Tile<1, RAJA::tile_fixed<j_block_sz>`` is used on the 
'j' loop. Note that we do not tile the 'k' loop, since the block size is one 
in that dimension. The other main difference with the previous block-stride loop
version is that we map iterations within each tile directly to threads in
a block; for example, using a ``RAJA::cuda_block_y_direct`` policy type
for the 'j' loop. RAJA *direct* policy types eliminate the block-stride looping,
which is not necessary here since we prescribe a block-size of 256 which 
fits within the thread-block size limitation of the CUDA device programming 
model.

For context and comparison, here is the same kernel implementation using
CUDA directly:

.. literalinclude:: ../../../../exercises/kernelintro-execpols_solution.cpp
   :start-after: _cuda_tensorinit_tiled_direct_start
   :end-before: _cuda_tensorinit_tiled_direct_end
   :language: C++

The ``nested_init`` device kernel used here is:

.. literalinclude:: ../../../../exercises/kernelintro-execpols_solution.cpp
   :start-after: _cuda_tensorinit_kernel_start
   :end-before: _cuda_tensorinit_kernel_end
   :language: C++

A few differences between the CUDA and RAJA-CUDA versions are worth noting. 
First, the CUDA version uses the CUDA ``dim3`` construct to express the 
threads-per-block and number of thread-blocks to use: i.e., the 
``nthreads_per_block`` and ``nblocks`` variable definitions. Note that
RAJA provides a macro ``RAJA_DIVIDE_CEILING_INT`` to perform the proper
arithmetic to calculate the number of blocks based on the size of the
tensor and the block size in each dimension. Second, the mapping of thread
identifiers to the (i,j,k) indices is explicit in the device kernel. Third,
an explicit check of the (i,j,k) values is required in the CUDA implementation 
to avoid addressing memory out-of-bounds; i.e., 
``if ( i < N && j < N && k < N )...``. The RAJA kernel variants set similar
definitions internally and **mask out indices that would be out-of-bounds.**

Lastly, we show the RAJA HIP variants of the kernel, which are semantically
identical to the RAJA CUDA variants.

The RAJA-HIP block-stride loop variant:

.. literalinclude:: ../../../../exercises/kernelintro-execpols_solution.cpp
   :start-after: _raja_tensorinit_hip_start
   :end-before: _raja_tensorinit_hip_end
   :language: C++

and the HIP fixed thread-block size, tiled, direct thread mapping version:

.. literalinclude:: ../../../../exercises/kernelintro-execpols_solution.cpp
   :start-after: _raja_tensorinit_hip_tiled_direct_start
   :end-before: _raja_tensorinit_hip_tiled_direct_end
   :language: C++

The only differences are that type names are changed to replace 'CUDA' types 
with 'HIP' types to use the RAJA HIP back-end.

.. note:: The C++ lambda expression representing the kernel inner loop body
          in a HIP GPU kernel requires the ``__device__`` annotation, the
          same as for CUDA.
