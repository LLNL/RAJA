.. ##
.. ## Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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
capability to import RAJA into your CMake project.

To use the configuration file, you can add the following command to your CMake
project::

  find_package(RAJA)

Then, pass the path of RAJA to CMake when you configure your code::

  cmake -DRAJA_DIR=<path-to-raja>/share/raja/cmake

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
