.. #######################################################################
.. #
.. # Copyright (c) 2016, Lawrence Livermore National Security, LLC.
.. #
.. # Produced at the Lawrence Livermore National Laboratory.
.. #
.. # All rights reserved.
.. #
.. # This source code cannot be distributed without permission and
.. # further review from Lawrence Livermore National Laboratory.
.. #
.. #######################################################################


===================================
Configuring and building the code
===================================

This section provides a brief discussion of how to configure and build
the RAJA code.

RAJA CMake System
-----------------

RAJA uses a simple CMake-based system, but you don't need to know much 
about CMake to build RAJA. It does require that you use a CMake version 
greater than 3.1. If needed, you can learn more about CMake and download
it at `<http://www.cmake.org>`_

An important thing to note is that our CMake setup does not allow
in-source builds so that you can easily configure and build for various
systems and compilers from the same source code.

Configuring the code can be done in a couple of ways depending on your needs.

**Use the provided configuration script.**

  If you have python installed on your system, you can execute a
  python helper script from the root directory of the RAJA source code
  to configure a build ::

    $ ./scripts/config-build.py

  This script will find and use a configuration file to initialize the
  CMake cache and create build and install directories in the RAJA root
  directory whose names contain the system type, compiler, and built type. 
  For example ::

    $ ls
    build-chaos-gnu-debug 
    install-chaos-gnu-debug 

  The configuration files are located in the 'host-config' directory. 
  The 'host-configs' directory contains configuration files for platforms
  and compilers most commonly used in the Livermore Computing Center at
  Lawrence Livermore National Laboratory. The contents of these files and
  how they are used in RAJA are described below. Note that new host 
  configuration cache files can be created by copying an existing one 
  to a new file and modifying the contents for your needs.

  If you run the python script with no arguments, it will configure the
  code using a default compiler and options for the platform you are on.
  The script accepts several options for

    * specifying the build and install directory paths and names
    * specifying the compiler
    * specifying the build type (Release, Debug, RelWithDebInfo)
    * specifying CMake options
    * selecting the host configuration file to initialize the CMake cache

  Script options can be displayed by running ::

    $ ./scripts/config-build.py --help

**Run CMake directly.**

  You can also run CMake directly. To configure the code using this method,
  create a build directory in the root directory and go into it ::

    $ my-awesome-build
    $ cd my-awesome-build

  Then, invoke CMake and specify the build type and cache file you want and
  the location of the top-level RAJA CMakeLists.txt file; for example ::

    $ cmake -DCMAKE_BUILD_TYPE=Debug -C ../host-configs/chaos/gnu_4_9_3.cmake ../

**Note that compiling with 'nvcc' on LC machines that have GPUs that support 
CUDA, you will have to load the CUDA module and set the host compiler.** For 
example, type these lines :: 

  $ module load cudatoolkit/7.5
  $ use gcc-4.9.3p

At least we know things work if you use this CUDA and host compiler 
combination.

Regardless of how you configure your build, you build the code by going into 
the build directory and typing ::

  $ make

If you want to also create an installation of the code, you can type ::

  $ make install

This will create 'include' and 'lib' directories in the install directory.



RAJA configuration options
---------------------------

The RAJA include directory 'include/RAJA' contains a header file 
called 'config.hxx.in' that is used to centralize all the configuration
options that RAJA supports in one location. Most RAJA constructs are 
parametrized to make it easy to try alternative implementation choices.

The items in the configuration header file are set when the code is 
configured. The results appear in the 'config.hxx' file which lives in 
the 'include/RAJA' directory in the build space. This file gets pulled into
all other RAJA header files so everything is consistent. The settings are 
controlled by the contents of the selected host configuration
file and the top-level RAJA 'CMakeLists.txt file'. For example, the file
associated with the Intel compiler on LLNL Linux platforms is: ::

  set(RAJA_COMPILER "RAJA_COMPILER_ICC" CACHE STRING "")

  set(CMAKE_C_COMPILER "/usr/local/bin/icc-16.0.109" CACHE PATH "")
  set(CMAKE_CXX_COMPILER "/usr/local/bin/icpc-16.0.109" CACHE PATH "")

  if(CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx -inline-max-total-size=20000 -inline-forceinline -ansi-alias -std=c++0x" CACHE STRING "")
  elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx -inline-max-total-size=20000 -inline-forceinline -ansi-alias -std=c++0x" CACHE STRING "")
  elseif(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -std=c++0x" CACHE STRING "")
  endif()

  set(RAJA_USE_OPENMP On CACHE BOOL "")
  set(RAJA_USE_CILK On CACHE BOOL "")

  set(RAJA_RANGE_ALIGN 4 CACHE INT "")
  set(RAJA_RANGE_MIN_LENGTH 32 CACHE INT "")
  set(RAJA_DATA_ALIGN 64 CACHE INT "")
  set(RAJA_COHERENCE_BLOCK_SIZE 64 CACHE INT "")

The first line sets a RAJA compiler variable that is used to control 
compiler-specific syntax for certain RAJA features. The next several 
commands in the file set the compiler and options for each build type. 
Next, programming model options, such as OpenMP, CilkPlus, CUDA, etc. are 
turned on or off. For example, the Intel compiler supports both OpenMP and 
CilkPlus; so those are turned on here. Finally, options for data alignment, 
index set range segments, and other things are set.

The CMakeLists.txt file in the top-level RAJA directory controls settings 
for other items that are not specific to a compiler. In that file, you will 
find variables to set RAJA options for: 

  * Floating-point type (e.g., double or float)
  * Pointer types (e.g., bare ptr, ptr with restrict, ptr classes, etc.)
  * Loop-level fault tolerance options
  * Timer options for examples
