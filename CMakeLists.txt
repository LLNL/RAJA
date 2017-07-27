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

# C is required for googletest to find Threads
project(RAJA LANGUAGES CXX C)

# Set version number
set(RAJA_VERSION_MAJOR 0)
set(RAJA_VERSION_MINOR 3)
set(RAJA_VERSION_PATCHLEVEL 0)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/thirdparty" ${CMAKE_MODULE_PATH})

# Build options
set(ENABLE_OPENMP On CACHE Bool "Build OpenMP support")
set(ENABLE_CUDA Off CACHE Bool "Build CUDA support")
set(ENABLE_COPY_HEADERS On CACHE Bool "")

option(ENABLE_TARGET_OPENMP "Build OpenMP on target device support" Off)
option(ENABLE_CLANG_CUDA "Use Clang's native CUDA support" Off)
set(RAJA_CUDA_ARCH "sm_35" CACHE STRING "Compute architecture to pass to CUDA builds")
option(ENABLE_CUB "Use cub for scans using CUDA" On)
option(ENABLE_TESTS "Build tests" On)
option(ENABLE_EXAMPLES "Build simple examples" On)
option(ENABLE_WARNINGS "Enable warnings as errors for CI" Off)
option(ENABLE_DOCUMENTATION "Build RAJA documentation" Off)
set(TEST_DRIVER "" CACHE STRING "driver used to wrap test commands")


if (NOT EXISTS ${PROJECT_SOURCE_DIR}/blt/SetupBLT.cmake)
  message(FATAL_ERROR "\
  The BLT submodule is not present. \
  If in a git repo run the following command:\n\
  git submodule init && git submodule update")
endif()

include(blt/SetupBLT.cmake)

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


set (raja_sources
  src/AlignedRangeIndexSetBuilders.cpp
  src/DepGraphNode.cpp
  src/LockFreeIndexSetBuilders.cpp
  src/MemUtils_CPU.cpp
  src/MemUtils_CUDA.cpp
  src/ThreadUtils_CPU.cpp)

set (raja_headers
  include/RAJA/RAJA.hpp
  include/RAJA/config.hpp.in
  include/RAJA/index
  include/RAJA/index/IndexSet.hpp
  include/RAJA/index/IndexSetBuilders.hpp
  include/RAJA/index/IndexSetUtils.hpp
  include/RAJA/index/IndexValue.hpp
  include/RAJA/index/ListSegment.hpp
  include/RAJA/index/RangeSegment.hpp
  include/RAJA/internal/DepGraphNode.hpp
  include/RAJA/internal/ForallNPolicy.hpp
  include/RAJA/internal/IndexArray.hpp
  include/RAJA/internal/Iterators.hpp
  include/RAJA/internal/LegacyCompatibility.hpp
  include/RAJA/internal/MemUtils_CPU.hpp
  include/RAJA/internal/RAJAVec.hpp
  include/RAJA/internal/ThreadUtils_CPU.hpp
  include/RAJA/internal/fault_tolerance.hpp
  include/RAJA/internal/Span.hpp
  include/RAJA/internal/type_traits.hpp
  include/RAJA/pattern/forall.hpp
  include/RAJA/pattern/forallN.hpp
  include/RAJA/pattern/permute.hpp
  include/RAJA/pattern/reduce.hpp
  include/RAJA/pattern/scan.hpp
  include/RAJA/pattern/tile.hpp
  include/RAJA/policy/MultiPolicy.hpp
  include/RAJA/policy/PolicyBase.hpp
  include/RAJA/policy/cuda.hpp
  include/RAJA/policy/fwd.hpp
  include/RAJA/policy/openmp.hpp
  include/RAJA/policy/cuda/MemUtils_CUDA.hpp
  include/RAJA/policy/cuda/forall.hpp
  include/RAJA/policy/cuda/forallN.hpp
  include/RAJA/policy/cuda/policy.hpp
  include/RAJA/policy/cuda/raja_cudaerrchk.hpp
  include/RAJA/policy/cuda/reduce.hpp
  include/RAJA/policy/cuda/scan.hpp
  include/RAJA/policy/cuda/fwd.hpp
  include/RAJA/policy/sequential.hpp
  include/RAJA/policy/openmp/forall.hpp
  include/RAJA/policy/openmp/forallN.hpp
  include/RAJA/policy/openmp/policy.hpp
  include/RAJA/policy/openmp/reduce.hpp
  include/RAJA/policy/openmp/scan.hpp
  include/RAJA/policy/openmp/fwd.hpp
  include/RAJA/policy/openmp/target_forall.hpp
  include/RAJA/policy/openmp/target_reduce.hpp
  include/RAJA/policy/simd.hpp
  include/RAJA/policy/sequential/forall.hpp
  include/RAJA/policy/sequential/policy.hpp
  include/RAJA/policy/sequential/reduce.hpp
  include/RAJA/policy/sequential/scan.hpp
  include/RAJA/policy/sequential/fwd.hpp
  include/RAJA/policy/simd/forall.hpp
  include/RAJA/policy/simd/policy.hpp
  include/RAJA/policy/simd/fwd.hpp
  include/RAJA/util/Layout.hpp
  include/RAJA/util/OffsetLayout.hpp
  include/RAJA/util/Operators.hpp
  include/RAJA/util/Permutations.hpp
  include/RAJA/util/PermutedLayout.hpp
  include/RAJA/util/Timer.hpp
  include/RAJA/util/View.hpp
  include/RAJA/util/defines.hpp
  include/RAJA/util/types.hpp
  include/RAJA/util/chai_support.hpp)

set (raja_depends)

if (ENABLE_OPENMP)
  set (raja_depends
    openmp)
endif()

if (ENABLE_CUDA)
  set (raja_depends
    cuda)
endif ()

blt_add_library(
  NAME RAJA
  SOURCES ${raja_sources}
  HEADERS ${raja_headers}
  HEADERS_OUTPUT_SUBDIR RAJA
  DEPENDS_ON ${raja_depends})

install(TARGETS RAJA 
  EXPORT RAJA
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib
)

install(EXPORT RAJA DESTINATION share/raja/cmake/)

#include_directories(${PROJECT_BINARY_DIR}/include/RAJA)

target_include_directories(RAJA PRIVATE ${PROJECT_BINARY_DIR}/include)

install(DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN *.hpp)
install(FILES ${PROJECT_BINARY_DIR}/include/RAJA/config.hpp
        DESTINATION "include/RAJA")

if(ENABLE_TESTS)
  add_subdirectory(test)
endif()

if(ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

if (ENABLE_DOCUMENTATION)
  add_subdirectory(docs)
endif ()
