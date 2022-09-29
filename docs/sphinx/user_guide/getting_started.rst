.. ##
.. ## Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##


.. _getting_started-label:

*************************
Getting Started With RAJA
*************************

This section will help get you up and running with RAJA quickly.

============
Requirements
============

The primary requirement for using RAJA is a C++14 standard compliant compiler.
Using various programming model back-ends requires that they be supported
by the compiler you chose. Available options and how to enable or disable 
them are described in :ref:`configopt-label`. To build RAJA in its most basic
form and use its simplest features, you will need:

- C++ compiler with C++14 support
- `CMake <https://cmake.org/>`_ version 3.14.5 or greater.


==================
Get the Code
==================

The RAJA project is hosted on `GitHub <https://github.com/LLNL/RAJA>`_.
To get the code, clone the repository into a local working space using
the command::

   $ git clone --recursive https://github.com/LLNL/RAJA.git

The ``--recursive`` argument above is needed to pull in RAJA dependencies as 
Git *submodules*. 

After running the clone command, a copy of the RAJA repository will reside in
a ``RAJA`` subdirectory where you ran the clone command. You will be on the 
``develop`` branch of RAJA, which is our default branch.

If you do not pass the ``--recursive`` argument to the ``git clone``
command, you can type the following commands after cloning::

  $ cd RAJA
  $ git submodule update --init --recursive

Either way, the end result is the same and you should be good to go.

.. note:: * If you switch branches in a RAJA repo, you may need to run the
            command ``git submodule update`` set the Git submodule versions to
            what is used by the new branch.
          * If the set of submodules in the new branch is different than the
            previous branch you were on, you may need to run the command
            ``git submodule update --init --recursive``.

.. _getting_started_depend-label:

==================
Dependencies
==================

RAJA has several dependencies that are required depending on how you want to
build and use it. The RAJA Git repository contains several submodules that
contain these dependencies.

Dependencies that are required to build the RAJA code are:

- `BLT build system <https://github.com/LLNL/blt>`_
- `Camp compiler agnostic metaprogramming library  <https://github.com/LLNL/camp>`_

Other dependencies that users should be aware of are:

- `CUB CUDA utilities library <https://github.com/NVlabs/cub>`_ is required for using the RAJA CUDA back-end.
- `rocPRIM HIP parallel primitives library <https://github.com/ROCmSoftwarePlatform/rocPRIM.git>`_ is required for using the RAJA HIP back-end.
- `Desul <https://github.com/desul/desul>`_ is required if you want to use Desul atomics in RAJA instead of our current default atomics. Note that we plan to switch over to Desul atomics exclusively at some point.

Additional discussion of these dependencies, with respect to building RAJA, is 
provided in :ref:`getting_started_build-label`. Other than that, you probably 
don't need to know much about them. If you are curious and want to know more, 
please click on the link to the library you want to know about above.

.. note:: You may want or need to use external versions of camp, CUB, or 
          rocPRIM instead of the RAJA submodules. To do so, you need to use
          CMake variables to pass a path to a valid installation of each 
          library. Specifically:

            * External camp: -Dcamp_DIR=<camp dir name>
            * External CUB: -DRAJA_ENABLE_EXTERNAL_CUB=On -DCUB_DIR=<CUB dir name>
            * External rocPRIM: -DRAJA_ENABLE_EXTERNAL_ROCPRIM=On -DROCPRIM_DIR=<rocPRIM dir name>

More information about configuring GPU builds with CUDA or HIP is provided
in :ref:`getting_started_build_gpu-label`

RAJA includes other submodule dependencies, which are used to support our 
Gitlab CI testing. These are described in the RAJA Developer Guide. 

.. _getting_started_build-label:

==================
Build and Install
==================

Building and installing RAJA can be very easy or more complicated, depending
on which features you want to use and how easy it is to use your system.

RAJA uses CMake to configure a build. A "bare bones" configuration, and build
and install looks like::

  $ mkdir build-dir && cd build-dir
  $ cmake -DCMAKE_INSTALL_PREFIX=/path/to/install ../
  $ make  (or make -j <N>)
  $ make install

.. note:: RAJA builds must be *out-of-source*. In particular, RAJA does not al
          low building in its source directory. You must create a build 
          directory and run CMake in it.

If you want to use a C++ compiler other than the default on your system, 
you need to pass a path to the compiler using the standard CMake option
``-DCMAKE_CXX_COMPILER=path/to/compiler``.

When you run CMake, it will generate output about the build environment 
(compiler and version, options, etc.). For a summary of RAJA configuration 
options, please see :ref:`configopt-label`.

After CMake successfully completes, you compile RAJA by executing the ``make``
command in the build directory::

  $ make

If you are on a multi-core system, you can compile in parallel by 
running ``make -j`` (to build with all available cores) or ``make -j N`` to 
build using N cores.

.. note:: RAJA is configured to build its tests, examples, and tutorial
          exercises by default. If you do not disable them with the 
          appropriate CMake option (please see :ref:`configopt-label`), 
          you can run them after the build completes to check if everything 
          is built properly.

          The easiest way to run the full set of RAJA tests is to type::

             $ make test

          in the build directory after the build completes.

          You can also run individual tests by invoking test 
          executables directly. They will be located in the ``test`` 
          subdirectory in the build space. RAJA tests use the 
          `Google Test framework <https://github.com/google/googletest>`_, 
          so you can also run and filter tests via Google Test commands.

          The source files for RAJA examples and exercises are located in 
          the ``RAJA/examples`` and ``RAJA/exercises`` directories, 
          respectively. When built, the executables for the examples and 
          exercises will be located in the ``bin`` subdirectory in the build 
          space. Feel free to experiment by editing the source files,
          recompiling, and running with your changes. 

.. _getting_started_build_gpu-label:

-----------------
GPU Builds
-----------------

CUDA
^^^^^^

To run RAJA code on NVIDIA GPUs, one typically must have a CUDA compiler 
installed on your system, in addition to a host code compiler. You may need 
to specify both when you run CMake. The host compiler is specified using the 
``CMAKE_CXX_COMPILER`` CMake variable as described earlier. The CUDA software
stack and compiler are specified using the following CMake options:

  * -DCUDA_TOOLKIT_ROOT_DIR=path/to/cuda/toolkit
  * -DCMAKE_CUDA_COMPILER=path/to/nvcc

When using the NVIDIA nvcc compiler for RAJA CUDA functionality, the variables:

  * CMAKE_CUDA_FLAGS_RELEASE
  * CMAKE_CUDA_FLAGS_DEBUG
  * CMAKE_CUDA_FLAGS_RELWITHDEBINFO

correspond to the standard CMake build types and are used to pass additional
compiler options to nvcc.

.. note:: When nvcc must pass options to the host compiler, the arguments
          can be included using these CMake variables. Host compiler
          options must be prepended with the `-Xcompiler` directive.

To set the CUDA compute architecture for the nvcc compiler, which should be
chosen based on the NVIDIA GPU hardware you are using, you can use the
``CUDA_ARCH`` CMake variable. For example, the CMake option 
``-DCUDA_ARCH=sm_60`` will tell the compiler to use the `sm_60` SASS 
architecture in its second stage of compilation. It will pick the PTX 
architecture to use in the first stage of compilation that is suitable for 
the SASS architecture you specify.

Alternatively, you may specify the PTX and SASS architectures, using
appropriate nvcc options in the ``CMAKE_CUDA_FLAGS_*`` variables.

.. note:: **RAJA requires a minimum CUDA architecture level of `sm_35` to use
          all supported CUDA features.** Mostly, the architecture level affects
          which RAJA CUDA atomic operations are available and how they are
          implemented inside RAJA. This is described in :ref:`feat-atomics-label`.

          * If you do not specify a value for ``CUDA_ARCH``, it will be set to
            `sm_35` by default and CMake will emit a status message 
            indicating this choice was made.

          * If you give a ``CUDA_ARCH`` value less than `sm_35` (e.g., `sm_30`),
            CMake will report this and stop processing.

Also, RAJA relies on the CUB CUDA utilities library for some CUDA functionality.
The CUB included in the CUDA toolkit is used by default, if available. This
is the case for CUDA version 11 and later. RAJA includes a CUB submodule that 
can be used with older versions of CUDA. To use an external CUB install 
provide the following option to CMake:
``-DRAJA_ENABLE_EXTERNAL_CUB=On -DCUB_DIR=<path/to/cub>``.

.. note:: It is important to note that the CUDA toolkit version of cub is
          required for compatibility with the CUDA toolkit version of thrust
          starting with CUDA toolkit version v11.0.0. So, if you build
          RAJA with CUDA version 11 or higher you should use the version of
          CUB contained in the CUDA toolkit version you are using to use 
          Thrust and be compatible with libraries that use Thrust.

.. note:: It is important to note that the version of Googletest that
          is used in RAJA version v0.11.0 or newer requires CUDA version
          9.2.x or newer when compiling with nvcc. Thus, if you build
          RAJA with CUDA enabled and want to also enable RAJA tests, you
          must use CUDA version 9.2.x or newer.

HIP
^^^^

To run RAJA code on AMD GPUs, one typically uses the ROCm compiler and tool 
chain (which can also be used to compile code for NVIDIA GPUs).

.. note:: RAJA requires version 3.5 or newer of the ROCm software stack to 
          use the RAJA HIP back-end.

Unlike CUDA, you do not specify a host compiler and a device compiler. 
Typical CMake options to use when building with a ROCm stack are:

  * -DROCM_ROOT_DIR=path/to/rocm
  * -DHIP_ROOT_DIR=path/to/hip
  * -DHIP_PATH=path/to/hip/binaries
  * -DCMAKE_CXX_COMPILER=path/to/rocm/compiler 

Additionally, you use the CMake variable ``CMAKE_HIP_ARCHITECTURES`` to set
the target compute architecture. For example::

  -DCMAKE_HIP_ARCHITECTURES=gfx908

RAJA relies on the rocPRIM HIP utilities library for some HIP
functionality. The rocPRIM included in the ROCm install is used by default if
available. RAJA includes a rocPRIM submodule that is used if it is not
available. To use an external rocPRIM install provide the following option to CMake:
``-DRAJA_ENABLE_EXTERNAL_ROCPRIM=On -DROCPRIM_DIR=<pat/to/rocPRIM>``.

.. note:: When using HIP and targeting NVIDIA GPUs, RAJA uses CUB instead of
          rocPRIM. In this case you must use an external CUB install using the
          CMake variables described in the CUDA section.

OpenMP
^^^^^^^

To use OpenMP target offload GPU execution, additional options may need to be
passed to the compiler. The variable ``OpenMP_CXX_FLAGS`` is used for this.
Option syntax follows the CMake *list* pattern. For example, to specify OpenMP 
target options for NVIDIA GPUs using a clang-based compiler, one may do
something like::

   cmake \
     ....
     -DOpenMP_CXX_FLAGS="-fopenmp;-fopenmp-targets=nvptx64-nvidia-cuda"

----------------------------------------
RAJA Example Build Configuration Files
----------------------------------------

The ``RAJA/scripts`` directory contains subdirectories with a variety of
build scripts we use to build and test RAJA on various platforms with
various compilers. These scripts pass files (*CMake cache files*) located in
the ``RAJA/host-configs`` directory to CMake using the '-C' option.
These files serve as useful examples of how to configure RAJA prior to
compilation.

----------------
Installing RAJA
----------------

To install RAJA as a library, run the following command in your build 
directory::

  $ make install

This will copy RAJA header files to the ``include`` directory and the RAJA
library will be installed in the ``lib`` directory you specified using the
``-DCMAKE_INSTALL_PREFIX`` CMake option.


======================
Learning to Use RAJA
======================

If you want to view and run a very simple RAJA example code, a good place to
start is located in the file: ``RAJA/examples/daxpy.cpp``. After building 
RAJA with the options you select, the executable for this code will reside 
in the file: ``<build-dir>/examples/bin/daxpy``. Simply type the name
of the executable in your build directory to run it; i.e.,::

  $ ./examples/bin/daxpy 

The ``RAJA/examples`` directory also contains many other RAJA example codes 
you can run and experiment with.

For an overview of all the main RAJA features, see :ref:`features-label`.
A full tutorial with a variety of examples showing how to use RAJA features
can be found in :ref:`tutorial-label`.
