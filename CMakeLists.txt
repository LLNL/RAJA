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

cmake_minimum_required (VERSION 3.0)

project(RAJA LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 14) # note, will decay automatically

# Set version number
set(RAJA_VERSION_MAJOR 0)
set(RAJA_VERSION_MINOR 1)
set(RAJA_VERSION_PATCHLEVEL 0)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules" ${CMAKE_MODULE_PATH})

# Build options
option(RAJA_ENABLE_OPENMP "Build OpenMP support" On)
option(RAJA_ENABLE_CUDA "Build CUDA support" Off)
option(RAJA_ENABLE_CILK "Build Cilk support" Off)
option(RAJA_ENABLE_TESTS "Build tests" On)
option(RAJA_ENABLE_EXAMPLES "Build simple examples" On)
option(RAJA_ENABLE_NESTED "Enable nested loop support" Off)

set(TEST_DRIVER "" CACHE STRING "driver used to wrap test commands")

# Setup basic CMake options
include(cmake/SetupBasics.cmake)
# Setup vendor-specific compiler flags
include(cmake/SetupCompilers.cmake)
# Find third-party packages
include(cmake/SetupPackages.cmake)
# Setup internal RAJA configuration options
include(cmake/SetupRajaConfig.cmake)

# Configure a header file with all the variables we found.
configure_file(${PROJECT_SOURCE_DIR}/include/RAJA/config.hxx.in
  ${PROJECT_BINARY_DIR}/include/RAJA/config.hxx)
configure_file(${PROJECT_SOURCE_DIR}/include/RAJA/RAJA-config.cmake.in
  ${PROJECT_BINARY_DIR}/include/RAJA/raja-config.cmake)
include_directories(${PROJECT_BINARY_DIR}/include/RAJA)
include_directories(${PROJECT_BINARY_DIR}/include)

include_directories(include)

install(DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN *.hxx)
install(FILES ${PROJECT_BINARY_DIR}/include/RAJA/config.hxx DESTINATION "include/RAJA")


add_subdirectory(src)

install(FILES ${PROJECT_BINARY_DIR}/include/RAJA/raja-config.cmake
  DESTINATION cmake/)

if(RAJA_ENABLE_TESTS)
  add_subdirectory(test)
endif()

if(RAJA_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

include(FindPkgConfig QUIET)
if(PKG_CONFIG_FOUND)
  # convert lists of link libraries into -lstdc++ -lm etc..
  foreach(LIB ${CMAKE_CXX_IMPLICIT_LINK_LIBRARIES} ${PLATFORM_LIBS} ${CUDA_LIBRARIES})
    set(PRIVATE_LIBS "${PRIVATE_LIBS} -l${LIB}")
  endforeach()
  foreach(INCDIR ${INCLUDE_DIRECTORIES} ${CUDA_INCLUDE_DIRS})
    set(PC_C_FLAGS "${PC_C_FLAGS} -I${INCDIR}")
  endforeach()
  if(RAJA_ENABLE_CUDA)
    foreach(FLAG ${RAJA_NVCC_FLAGS})
      set(PC_C_FLAGS "${PC_C_FLAGS} ${FLAG}")
    endforeach()
  else()
    foreach(FLAG ${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS})
      set(PC_C_FLAGS "${PC_C_FLAGS} ${FLAG}")
    endforeach()
  endif()
  # Produce a pkg-config file for linking against the shared lib
  configure_file("pkg-config/RAJA.pc.in" "RAJA.pc" @ONLY)
  install(FILES       "${CMAKE_CURRENT_BINARY_DIR}/RAJA.pc"
          DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig")
endif()
