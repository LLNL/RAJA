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

The primary requirement for using RAJA is a C++14 compliant compiler.
Accessing various programming model back-ends requires that they be supported
by the compiler you chose. Available options and how to enable or disable 
them are described in :ref:`configopt-label`. To build RAJA in its most basic
form and use its simplest features:

- C++ compiler with C++14 support
- `CMake <https://cmake.org/>`_ version 3.14.5 or greater.


==================
Get the Code
==================

The RAJA project is hosted on `GitHub <https://github.com/LLNL/RAJA>`_.
To get the code, clone the repository into a local working space using
the command::

   $ git clone --recursive https://github.com/LLNL/RAJA.git

The ``--recursive`` argument above is needed to pull in necessary RAJA
dependencies as Git *submodules*. Current RAJA dependencies are:

- `BLT build system <https://github.com/LLNL/blt>`_
- `Camp compiler agnostic metaprogramming library  <https://github.com/LLNL/camp>`_
- `CUB CUDA utilities library <https://github.com/NVlabs/cub>`_
- `rocPRIM HIP parallel primitives library <https://github.com/ROCmSoftwarePlatform/rocPRIM.git>`_

You probably don't need to know much about these other projects to start
using RAJA. But, if you want to know more about them, click on the links above.

After running the clone command, a copy of the RAJA repository will reside in
a ``RAJA`` subdirectory where you ran the clone command. You will be on the 
``develop`` branch of RAJA, which is our default branch.

If you do not pass the ``--recursive`` argument to the ``git clone``
command, you can type the following commands after cloning::

  $ cd RAJA
  $ git submodule init
  $ git submodule update

Either way, the end result is the same and you should be good to go.

.. note:: Any time you switch branches in RAJA, you need to re-run the
          'git submodule update' command to set the Git submodules to
          what is used by the new branch.

==================
Build and Install
==================

Building and installing RAJA can be very easy or more complicated, depending
on which features you want to use and how easy it is to use your system.

--------------
Building RAJA
--------------

RAJA uses CMake to configure a build. A "bare bones" configuration looks like::

  $ mkdir build-dir && cd build-dir
  $ cmake -DCMAKE_INSTALL_PREFIX=/path/to/install ../

.. note:: * RAJA requires a minimum CMake version of 3.14.5.
          * Builds must be *out-of-source*.  RAJA does not allow building in
            the source directory, so you must create a build directory and
            run CMake in it.

When you run CMake, it will generate output about the build environment 
(compiler and version, options, etc.). Some RAJA features, 
like OpenMP support are enabled by default if, for example, the compiler 
supports OpenMP. These can be disabled if desired. For a summary of 
RAJA configuration options, please see :ref:`configopt-label`.

After CMake successfully completes, you compile RAJA by executing the ``make``
command in the build directory; i.e.,::

  $ make

If you have access to a multi-core system, you can compile in parallel by 
running ``make -j`` (to build with all available cores) or ``make -j N`` to 
build using N cores.

.. note:: * RAJA is configured to build its unit tests by default. If you do not
            disable them with the appropriate CMake option (please see
            :ref:`configopt-label`), you can run them after the build completes
            to check if everything is built properly.

            The easiest way to run the full set of RAJA tests is to type::

               $ make test

            in the build directory after the build completes.

            You can also run individual tests by invoking test 
            executables directly. They will be located in the ``test`` 
            subdirectory in the build space directory. RAJA tests use the 
            `Google Test framework <https://github.com/google/googletest>`_, 
            so you can also run tests via Google Test commands.

          * RAJA also contains example and tutorial exercise 
            programs you can run if you wish. Similar to the RAJA tests, 
            the examples and exercises are built by default and can be
            disabled with CMake options (see :ref:`configopt-label`). The 
            source files for these are located in the ``RAJA/examples`` and 
            ``RAJA/exercises`` directories, respectively. When built, the
            executables for the examples and exercises will be located in
            the ``bin`` subdirectory in the build space directory. Feel free to 
            experiment by editing the source files and recompiling.

.. _build-external-tpl-label:

.. note:: You may use externally-supplied versions of the camp, CUB, and rocPRIM
          libraries with RAJA if you wish. To do so, pass the following
          options to CMake:
            * External camp: -DEXTERNAL_CAMP_SOURCE_DIR=<camp dir name>
            * External CUB: -DRAJA_ENABLE_EXTERNAL_CUB=On -DCUB_DIR=<CUB dir name>
            * External rocPRIM: -DRAJA_ENABLE_EXTERNAL_ROCPRIM=On
                                -DROCPRIM_DIR=<rocPRIM dir name>

-----------------
GPU Builds, etc.
-----------------

CUDA
^^^^^^

To run RAJA code on NVIDIA GPUs, one typically must have a CUDA compiler 
installed on your system, in addition to a host code compiler. You may need 
to specify both when you run CMake. The host compiler is specified using the 
``CMAKE_CXX_COMPILER`` CMake variable. The CUDA compiler is specified with
the ``CMAKE_CUDA_COMPILER`` variable.

When using the NVIDIA nvcc compiler for RAJA CUDA functionality, the variables:

  * CMAKE_CUDA_FLAGS_RELEASE
  * CMAKE_CUDA_FLAGS_DEBUG
  * CMAKE_CUDA_FLAGS_RELWITHDEBINFO

which corresponding to the standard CMake build types are used to pass flags
to nvcc.

.. note:: When nvcc must pass options to the host compiler, the arguments
          can be included using these CMake variables. Host compiler
          options must be prepended with the `-Xcompiler` directive.

To set the CUDA compute architecture for the nvcc compiler, which should be
chosen based on the NVIDIA GPU hardware you are using, you can use the
``CUDA_ARCH`` CMake variable. For example, the CMake option::

  -DCUDA_ARCH=sm_60

will tell the compiler to use the `sm_60` SASS architecture in its second
stage of compilation. It will pick the PTX architecture to use in the first
stage of compilation that is suitable for the SASS architecture you specify.

Alternatively, you may specify the PTX and SASS architectures, using
appropriate nvcc options in the ``CMAKE_CUDA_FLAGS_*`` variables.

.. note:: **RAJA requires a minimum CUDA architecture level of `sm_35` to use
          all supported CUDA features.** Mostly, the architecture level affects
          which RAJA CUDA atomic operations are available and how they are
          implemented inside RAJA. This is described in :ref:`atomics-label`.

          * If you do not specify a value for ``CUDA_ARCH``, it will be set to
            `sm_35` by default and CMake will emit a status message 
            indicatting this choice was made.

          * If you give a ``CUDA_ARCH`` value less than `sm_35` (e.g., `sm_30`),
            CMake will report this and stop processing.

Also, RAJA relies on the CUB CUDA utilities library for some CUDA functionality.
The CUB included in the CUDA toolkit is used by default if available. RAJA
includes a CUB submodule that is used if it is not available. To use
an external CUB install provide the following option to CMake:
``-DRAJA_ENABLE_EXTERNAL_CUB=On -DCUB_DIR=<pat/to/cub>``.

.. note:: **It is important to note that the CUDA toolkit version of cub is
          required for compatibility with the CUDA toolkit version of thrust
          starting with CUDA toolkit version v11.0.0. So, if you build
          RAJA with CUDA version 11 or higher you must use the CUDA
          toolkit version of CUB to use Thrust and be compatible with libraries
          that use Thrust.

          *It is important to note that the version of Googletest that
          is used in RAJA version v0.11.0 or newer requires CUDA version
          9.2.x or newer when compiling with nvcc. Thus, if you build
          RAJA with CUDA enabled and want to also enable RAJA tests, you
          must use CUDA version 9.2.x or newer.

HIP
^^^^

To run RAJA code on AMD GPUs, one typically uses the HIP compiler and tool 
chain (which can also be used to compile code for NVIDIA GPUs).

.. note:: RAJA requires version 3.5 or newer of the ROCm software stack to 
          use the RAJA HIP back-end.

Also, RAJA relies on the rocPRIM HIP utilities library for some HIP
functionality. The rocPRIM included in the ROCm install is used by default if
available. RAJA includes a rocPRIM submodule that is used if it is not
available. To use an external rocPRIM install provide the following option to CMake:
``-DRAJA_ENABLE_EXTERNAL_ROCPRIM=On -DROCPRIM_DIR=<pat/to/rocPRIM>``.

.. note:: When using HIP and targeting NVIDIA GPUs RAJA uses CUB instead of
          rocPRIM. In this case you must use an external CUB install using the
          CMake variables described in the CUDA section.

OpenMP
^^^^^^^

To use OpenMP target offlad GPU execution, additional options may need to be
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
