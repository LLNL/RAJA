###############################################################################
# Copyright (c) 2016, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-689114
#
# All rights reserved.
#
# This file is part of RAJA.
#
# For additional details, please also read RAJA/LICENSE.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the disclaimer below.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################

cmake_minimum_required (VERSION 3.3)

project(RAJA LANGUAGES CXX C)

# Set version number
set(RAJA_VERSION_MAJOR 0)
set(RAJA_VERSION_MINOR 2)
set(RAJA_VERSION_PATCHLEVEL 5)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/thirdparty" ${CMAKE_MODULE_PATH})

# Build options
option(RAJA_ENABLE_OPENMP "Build OpenMP support" On)
option(RAJA_ENABLE_CUDA "Build CUDA support" Off)
option(RAJA_ENABLE_CLANG_CUDA "Use Clang's native CUDA support" Off)
set(RAJA_CUDA_ARCH "sm_35" CACHE STRING "Compute architecture to pass to CUDA builds")
option(RAJA_ENABLE_CUB "Use cub for scans using CUDA" On)
option(RAJA_ENABLE_CILK "Build Cilk support" Off)
option(RAJA_ENABLE_TESTS "Build tests" On)
option(RAJA_ENABLE_EXAMPLES "Build simple examples" On)
option(RAJA_ENABLE_NESTED "Enable nested loop support" Off)
option(RAJA_ENABLE_WARNINGS "Enable warnings as errors for CI" Off)
option(RAJA_ENABLE_DOCUMENTATION "Build RAJA documentation" Off)

set(TEST_DRIVER "" CACHE STRING "driver used to wrap test commands")

# Setup basic CMake options
include(cmake/SetupBasics.cmake)
# Setup vendor-specific compiler flags
include(cmake/SetupCompilers.cmake)
# Find third-party packages
include(cmake/SetupPackages.cmake)
# Setup internal RAJA configuration options
include(cmake/SetupRajaConfig.cmake)
# Macros for building executables and libraries
include (cmake/RAJAMacros.cmake)
# Sanity check for compiler compatibility
include (cmake/CompilerCompatibility.cmake)

include_directories(${PROJECT_BINARY_DIR}/include/RAJA)
include_directories(${PROJECT_BINARY_DIR}/include)

include_directories(include)

install(DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN *.hpp)
install(FILES ${PROJECT_BINARY_DIR}/include/RAJA/config.hpp
        DESTINATION "include/RAJA")

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
