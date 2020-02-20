.. ##
.. ## Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/COPYRIGHT file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _using-raja-label:

******************************
Using RAJA in Your Application
******************************

Using RAJA in an application requires two things: ensuring the header files
are visible, and linking against the RAJA library. We maintain a 
`RAJA Template Project <https://github.com/LLNL/RAJA-project-template>`_
shows how to use RAJA in a CMake project, either as a Git submodule or
as an externally installed library that you link your application against.

It is important to ensure that you pass in compiler options in your project
that match the options that you compiled RAJA with. Specifically, if you built
RAJA with `-DENABLE_OPENMP=On` then you need to add a flag like `-fopenmp` when
compiling any files in your project that include RAJA.

Similarly, if you built RAJA with `-DENABLE_CUDA=On` then you need to pass in
some extra compile flags, and also ensure you use `nvcc` to compile any files
in your project that include RAJA. The required flags are:

- `-std=c++11`
- `-x cu ` required if your file extensions are not .cu
- `--expt-extended-lambda` 
- `-arch=smXX` where `smXX` is one of: 

========================
CMake Configuration File
========================

As part of the RAJA installation, we provide a ``RAJA-config.cmake`` file. If
your application uses CMake, this can be used with CMake's find_package
capability to import RAJA into your CMake project.

To use the configuration file, you can add the following command to your CMake
project::

  find_package(RAJA)

Then, pass the path of RAJA to CMake when you configure your code::

  cmake -DRAJA_DIR=<path-to-raja>/share/raja/cmake

The ``RAJA-config.cmake`` file provides a ``RAJA`` target, that can be used
natively by CMake to add a dependency on RAJA. For example::

  add_executable(my-app.exe
                 my-app.cpp)

  target_link_libraries(my-app.exe PUBLIC RAJA)

  ===================
  Using RAJA with make
  ===================

  If your project uses Make, then you need to 
