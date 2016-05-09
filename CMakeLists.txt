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
# For additional details, please also read raja/README-license.txt.
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

cmake_minimum_required (VERSION 2.8)

project(RAJA LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules" ${CMAKE_MODULE_PATH})

enable_testing()

# Set version number
set(RAJA_VERSION_MAJOR 1)
set(RAJA_VERSION_MINOR 0)
set(RAJA_VERSION_PATCHLEVEL 0)

# Setup some defaults if no host-config used
if(NOT RAJA_HOST_CONFIG_LOADED)
# Default build type
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
  endif()

  # Default constants
  if(NOT RAJA_RANGE_ALIGN)
    set(RAJA_RANGE_ALIGN 4)
  endif()
  if(NOT RAJA_RANGE_MIN_LENGTH)
    set(RAJA_RANGE_MIN_LENGTH 32)
  endif()
  if(NOT RAJA_DATA_ALGN)
    set(RAJA_DATA_ALIGN 64)
  endif()
  if(NOT RAJA_COHERENCE_BLOCK_SIZE)
    set(RAJA_COHERENCE_BLOCK_SIZE 64)
  endif()

  # Set up RAJA_COMPILER
  if (CMAKE_CXX_COMPILER_ID MATCHES Clang)
    set(RAJA_COMPILER "RAJA_COMPILER_CLANG")
  elseif (CMAKE_CXX_COMPILER_ID MATCHES GNU)
    set(RAJA_COMPILER "RAJA_COMPILER_GNU")
  elseif (CMAKE_CXX_COMPILER_ID MATCHES Intel)
    set(RAJA_COMPILER "RAJA_COMPILER_INTEL")
  endif()

  # Use OpenMP by default
  set(RAJA_USE_OPENMP On)
endif()

# We need to find the packages _after_ the CMake starts up, so this isn't in
# a host-config file.
if (RAJA_USE_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    list(APPEND RAJA_NVCC_FLAGS -Xcompiler ${OpenMP_CXX_FLAGS})
    message(STATUS "OpenMP Enabled")
  else()
    message(WARNING "OpenMP NOT FOUND")
    set(RAJA_USE_OPENMP Off)
  endif()
endif()

if (RAJA_USE_CUDA)
  find_package(CUDA)
  if(CUDA_FOUND)
    message(STATUS "CUDA")
    set (CUDA_NVCC_FLAGS ${RAJA_NVCC_FLAGS})
    set (CUDA_PROPAGATE_HOST_FLAGS OFF)
    include_directories(${CUDA_INCLUDE_DIRS})
  endif()
endif()

# Used for timing
find_library(RT_LIBRARIES rt)
#set(RT_LIBRARIES "")

#
# Configure all variables in the code 
#

## Floating point options
set(RAJA_FP "RAJA_USE_DOUBLE")
#set(RAJA_FP "RAJA_USE_FLOAT")

## Pointer options
if (RAJA_USE_CUDA)
  set(RAJA_PTR "RAJA_USE_BARE_PTR")
else ()
  set(RAJA_PTR "RAJA_USE_RESTRICT_PTR")
endif()
#set(RAJA_USE_BARE_PTR ON)
#set(RAJA_USE_RESTRICT_PTR OFF)
#set(RAJA_USE_RESTRICT_ALIGNED_PTR OFF)
#set(RAJA_USE_PTR_CLASS OFF)

## Fault tolerance options
set(RAJA_USE_FT OFF)
set(RAJA_REPORT_FT OFF)

## Timer options
add_definitions(-DRAJA_USE_GETTIME)
#add_definitions(-DRAJA_USE_CLOCK)
#add_definitions(-DRAJA_USE_CYCLE)

# Configure a header file with all the variables we found.
configure_file(${PROJECT_SOURCE_DIR}/include/RAJA/config.hxx.in
  ${PROJECT_BINARY_DIR}/include/RAJA/config.hxx)
include_directories(${PROJECT_BINARY_DIR}/include/RAJA)
include_directories(${PROJECT_BINARY_DIR}/include)

include_directories(include)

install(DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN *.hxx)
install(FILES ${PROJECT_BINARY_DIR}/include/RAJA/config.hxx DESTINATION "include/RAJA")


if (RAJA_USE_CUDA)
cuda_add_library(RAJA
    src/AlignedRangeIndexSetBuilders.cxx
    src/DepGraphNode.cxx
    src/IndexSet.cxx
    src/ListSegment.cxx
    src/LockFreeIndexSetBuilders.cxx
    src/MemUtils_CPU.cxx
    src/MemUtils_CUDA.cxx
    src/RangeSegment.cxx
    src/ThreadUtils_CPU.cxx
  )
else()
add_library(RAJA
    src/AlignedRangeIndexSetBuilders.cxx
    src/DepGraphNode.cxx
    src/IndexSet.cxx
    src/ListSegment.cxx
    src/LockFreeIndexSetBuilders.cxx
    src/MemUtils_CPU.cxx
    src/MemUtils_CUDA.cxx
    src/RangeSegment.cxx
    src/ThreadUtils_CPU.cxx
  )
endif()

install(TARGETS RAJA ARCHIVE DESTINATION lib)

set(BUILD_TESTS On CACHE STRING "Build test applications")
if(BUILD_TESTS)
  add_subdirectory(test)
endif()
