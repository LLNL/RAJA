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

if (RAJA_ENABLE_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    list(APPEND RAJA_NVCC_FLAGS -Xcompiler ${OpenMP_CXX_FLAGS})
    message(STATUS "OpenMP Enabled")
  else()
    message(WARNING "OpenMP NOT FOUND")
    set(RAJA_ENABLE_OPENMP Off)
  endif()
endif()

if (RAJA_ENABLE_CLANG_CUDA)
  set(RAJA_ENABLE_CUDA On)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif ()

if (RAJA_ENABLE_CUDA)
  find_package(CUDA)
  if(CUDA_FOUND)
    message(STATUS "CUDA Enabled")
    set (CUDA_NVCC_FLAGS ${RAJA_NVCC_FLAGS})
    set (CUDA_PROPAGATE_HOST_FLAGS OFF)
    include_directories(${CUDA_INCLUDE_DIRS})
  endif()

  if (RAJA_ENABLE_CUB)

    find_package(CUB)

    if (CUB_FOUND)
      include_directories(${CUB_INCLUDE_DIRS})
    else()
      message(WARNING "Using deprecated Thrust backend for CUDA scans.\n
  Please set CUB_DIR for better scan performance.")
      set(RAJA_ENABLE_CUB False)
    endif()
  endif()
endif()


if (RAJA_ENABLE_TESTS)
  include(ExternalProject)
  # Set default ExternalProject root directory
  SET_DIRECTORY_PROPERTIES(PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/tpl)

  ExternalProject_Add(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG release-1.7.0
      CMAKE_ARGS                
          -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
          -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      INSTALL_COMMAND ""
      LOG_DOWNLOAD ON
      LOG_CONFIGURE ON
      LOG_BUILD ON)

  ExternalProject_Get_Property(googletest source_dir)
  include_directories(${source_dir}/include)

  ExternalProject_Get_Property(googletest binary_dir)
  add_library(gtest      UNKNOWN IMPORTED)
  add_library(gtest_main UNKNOWN IMPORTED)

  if ( UNIX )
    set_target_properties(gtest PROPERTIES
      IMPORTED_LOCATION ${binary_dir}/libgtest.a
    )
    set_target_properties(gtest_main PROPERTIES
      IMPORTED_LOCATION ${binary_dir}/libgtest_main.a
    )
  elseif( WIN32 )
    set_target_properties(gtest PROPERTIES
      IMPORTED_LOCATION ${binary_dir}/${CMAKE_BUILD_TYPE}/gtest.lib
    )
    set_target_properties(gtest_main PROPERTIES
      IMPORTED_LOCATION ${binary_dir}/${CMAKE_BUILD_TYPE}/gtest_main.lib
    )
  endif ()
  add_dependencies(gtest      googletest)
  add_dependencies(gtest_main googletest)

  # GoogleTest requires threading
  find_package(Threads)

  enable_testing()
endif ()

if (RAJA_ENABLE_DOCUMENTATION)
  find_package(Sphinx)
  find_package(Doxygen)
endif ()
