#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#
#=== Usage ===================================================================
# This file allows RAJA to be automatically detected by other libraries
# using CMake.  To build with RAJA, you can do one of two things:
#
#   1. Set the RAJA_DIR environment variable to the root of the Caliper
#      installation.  If you loaded RAJA through a dotkit, this may already
#      be set, and RAJA will be autodetected by CMake.
#
#   2. Configure your project with this option:
#      -DRAJA_DIR=<RAJA install prefix>/share/
#
# If you have done either of these things, then CMake should automatically find
# and include this file when you call find_package(RAJA) from your
# CMakeLists.txt file.
#
#=== Components ==============================================================
#
# To link against these, just do, for example:
#
#   find_package(RAJA REQUIRED)
#   add_executable(foo foo.c)
#   target_link_libraries(foo RAJA)
#
# That's all!
#
if (NOT RAJA_CONFIG_LOADED)
  set(RAJA_CONFIG_LOADED TRUE)

  # Install layout
  set(RAJA_INSTALL_PREFIX /g/g16/hornung1/AASD/RAJA-repo/raja-WORK/RAJA/install_lc_toss3-gcc-8.1.0)
  set(RAJA_INCLUDE_DIR    /g/g16/hornung1/AASD/RAJA-repo/raja-WORK/RAJA/install_lc_toss3-gcc-8.1.0/include)
  set(RAJA_LIB_DIR        /g/g16/hornung1/AASD/RAJA-repo/raja-WORK/RAJA/install_lc_toss3-gcc-8.1.0/lib)
  set(RAJA_CMAKE_DIR      /g/g16/hornung1/AASD/RAJA-repo/raja-WORK/RAJA/install_lc_toss3-gcc-8.1.0/share/raja/cmake)

  # Includes needed to use RAJA
  set(RAJA_INCLUDE_PATH /g/g16/hornung1/AASD/RAJA-repo/raja-WORK/RAJA/install_lc_toss3-gcc-8.1.0/include)
  set(RAJA_LIB_PATH     /g/g16/hornung1/AASD/RAJA-repo/raja-WORK/RAJA/install_lc_toss3-gcc-8.1.0/lib)
  set(RAJA_COMPILE_FLAGS "     -Wall -Wextra ")
  set(RAJA_NVCC_FLAGS )

  set(RAJA_RT_LIBRARIES "")

  set(RAJA_TIMER_TYPE    )
  set(ENABLE_CUDA   Off)
  set(ENABLE_FT     OFF)
  set(ENABLE_OPENMP On)
  set(ENABLE_TARGET_OPENMP OFF)
  set(ENABLE_TESTS  ON)
  set(RAJA_REPORT_FT     OFF)
  set(RAJA_USE_COMPLEX   OFF)
  set(RAJA_USE_DOUBLE    OFF)
  set(RAJA_USE_FLOAT     OFF)
  # Library targets imported from file
  include(/g/g16/hornung1/AASD/RAJA-repo/raja-WORK/RAJA/install_lc_toss3-gcc-8.1.0/share/raja/cmake/RAJA.cmake)
endif()
