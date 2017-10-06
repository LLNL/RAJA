###############################################################################
# Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-689114
#
# All rights reserved.
#
# This file is part of RAJA.
#
# For details about use and distribution, please read RAJA/LICENSE.
#
###############################################################################

cmake_minimum_required (VERSION 3.3)

# C is required for googletest to find Threads
project(RAJA LANGUAGES CXX C)

# Set version number
set(RAJA_VERSION_MAJOR 0)
set(RAJA_VERSION_MINOR 3)
set(RAJA_VERSION_PATCHLEVEL 1)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/thirdparty" ${CMAKE_MODULE_PATH})

# Build options
set(RAJA_CXX_STANDARD_FLAG "default" CACHE STRING "Specific c++ standard flag to use, default attempts to autodetect the highest available")
option(RAJA_ENABLE_OPENMP "Build OpenMP support" On)
option(RAJA_ENABLE_TBB "Build TBB support" On)
option(RAJA_ENABLE_TARGET_OPENMP "Build OpenMP on target device support" Off)
option(RAJA_ENABLE_CUDA "Build CUDA support" Off)
option(RAJA_ENABLE_CLANG_CUDA "Use Clang's native CUDA support" Off)
set(RAJA_CUDA_ARCH "sm_35" CACHE STRING "Compute architecture to pass to CUDA builds")
option(RAJA_ENABLE_CUB "Use cub for scans using CUDA" On)
option(RAJA_ENABLE_TESTS "Build tests" On)
option(RAJA_ENABLE_EXAMPLES "Build simple examples" On)
option(RAJA_ENABLE_NESTED "Enable nested loop support" Off)
option(RAJA_ENABLE_MODULES "Enable modules in supporting compilers (clang)" On)
option(RAJA_ENABLE_WARNINGS "Enable warnings as errors for CI" Off)
option(RAJA_ENABLE_DOCUMENTATION "Build RAJA documentation" Off)

option(RAJA_ENABLE_COVERAGE "Enable coverage (only supported with GCC)" Off)

set(TEST_DRIVER "" CACHE STRING "driver used to wrap test commands")

# Setup basic CMake options
include(cmake/SetupBasics.cmake)
# Find third-party packages
include(cmake/SetupPackages.cmake)
# Setup vendor-specific compiler flags
include(cmake/SetupCompilers.cmake)
# Setup internal RAJA configuration options
include(cmake/SetupRajaConfig.cmake)
# Macros for building executables and libraries
include (cmake/RAJAMacros.cmake)
# Sanity check for compiler compatibility
include (cmake/CompilerCompatibility.cmake)

include_directories(include)
include_directories(include/RAJA/external)

include_directories(${PROJECT_BINARY_DIR}/include)

install(DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN *.hpp)

install(FILES
  ${PROJECT_BINARY_DIR}/include/RAJA/config.hpp
  include/RAJA/module.modulemap
  include/RAJA/module.private.modulemap
  DESTINATION "include/RAJA/")

add_subdirectory(src)

if(RAJA_ENABLE_TESTS)
  add_subdirectory(test)
endif()

if(RAJA_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

if (RAJA_ENABLE_DOCUMENTATION)
  add_subdirectory(docs)
endif ()

if(RAJA_ENABLE_APPLICATIONS)
  if (NOT EXISTS ${PROJECT_SOURCE_DIR}/extra/llnl-raja-proxies/CMakeLists.txt)
    message(STATUS "Cloning RAJA proxy applications...")
    execute_process(COMMAND git clone https://github.com/LLNL/RAJA-examples.git ${PROJECT_SOURCE_DIR}/extra/llnl-raja-proxies)
  endif()
  add_subdirectory(extra/llnl-raja-proxies)
endif()

if(RAJA_ENABLE_PERFSUITE)
  if (NOT EXISTS ${PROJECT_SOURCE_DIR}/extra/performance/CMakeLists.txt)
    message(STATUS "Cannot find performance suite")
    message(STATUS "Clone it to ./extra/performance")
  endif()
  add_subdirectory(extra/performance)
endif()
