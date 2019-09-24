.. ##
.. ## Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
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
are visible, and linking against the RAJA library.

========================
CMake Configuration File
========================

As part of the RAJA installation, we provide a ``RAJA-config.cmake`` file. If
your application uses CMake, this can be used with CMake's find_package
capability to import RAJA into your CMake project. RAJA also relies on the
``camp`` library. This is bundled with the RAJA repository, and will install a
``camp-config.cmake`` file that also works with CMake's find_package.

To use the configuration files, you can add the following command to your CMake
project::

  find_package(camp)
  find_package(RAJA)

Then, pass the path of RAJA and camp to CMake when you configure your code::

  cmake -DRAJA_DIR=<path-to-raja>/share/raja/cmake -Dcamp_DIR=<path-to-raja>/lib/camp/cmake ..

The ``RAJA-config.cmake`` file provides the following variables:

======================   ===================================
Variable                 Default
======================   ===================================
``RAJA_INCLUDE_DIR``     Include directory for RAJA headers.
``RAJA_LIB_DIR``         Library directory for RAJA.
``RAJA_COMPILE_FLAGS``   C++ flags used to compile RAJA.
``RAJA_NVCC_FLAGS``      CUDA flags used to compile RAJA.
======================   ===================================

It also provides the ``RAJA`` target, that can be used natively by CMake to add
a dependency on RAJA. For example::

  add_executable(my-app.exe
                 my-app.cpp)

  target_link_libraries(my-app.exe RAJA)

  target_include_directories(my-app.exe ${RAJA_INCLUDE_DIR}
