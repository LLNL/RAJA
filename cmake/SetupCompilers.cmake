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

if("cxx_std_17" IN_LIST CMAKE_CXX_KNOWN_FEATURES)
  set(CMAKE_CXX_STANDARD 17)
elseif("cxx_std_14" IN_LIST CMAKE_CXX_KNOWN_FEATURES)
  set(CMAKE_CXX_STANDARD 14)
endif()

set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0" CACHE STRING "")

if (RAJA_ENABLE_WARNINGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
    message(FATAL_ERROR "RAJA requires GCC 4.9 or greater!")
  endif ()
  if (RAJA_ENABLE_COVERAGE)
    if(NOT RAJA_ENABLE_CUDA)
      message(INFO "Coverage analysis enabled")
      set(CMAKE_CXX_FLAGS "-coverage ${CMAKE_CXX_FLAGS}")
      set(CMAKE_EXE_LINKER_FLAGS "-coverage ${CMAKE_EXE_LINKER_FLAGS}")
    endif()
  endif()
endif()

set(RAJA_COMPILER "RAJA_COMPILER_${CMAKE_CXX_COMPILER_ID}")

if ( MSVC )
  if (NOT BUILD_SHARED_LIBS)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
  endif()
endif()

if (RAJA_ENABLE_CUDA)
  # When we require cmake 3.8+, replace this with setting CUDA_STANDARD
  if(CUDA_VERSION_MAJOR GREATER "8")
    set(RAJA_NVCC_STD "c++14")
  else()
    set(RAJA_NVCC_STD "c++11")
  endif()
  if(CMAKE_BUILD_TYPE MATCHES Release)
    set(RAJA_NVCC_FLAGS -O2; -restrict; -arch ${RAJA_CUDA_ARCH}; -std ${RAJA_NVCC_STD}; --expt-extended-lambda; -ccbin; ${CMAKE_CXX_COMPILER} CACHE LIST "")
  elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
    set(RAJA_NVCC_FLAGS -g; -G; -O2; -restrict; -arch ${RAJA_CUDA_ARCH}; -std  ${RAJA_NVCC_STD}; --expt-extended-lambda; -ccbin ${CMAKE_CXX_COMPILER} CACHE LIST "")
  else(CMAKE_BUILD_TYPE MATCHES Debug)
    set(RAJA_NVCC_FLAGS -g; -G; -O0; -restrict; -arch ${RAJA_CUDA_ARCH}; -std  ${RAJA_NVCC_STD}; --expt-extended-lambda; -ccbin ${CMAKE_CXX_COMPILER} CACHE LIST "")
  endif()
  if(RAJA_ENABLE_COVERAGE)
    if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
      message(INFO "Coverage analysis enabled")
      set(RAJA_NVCC_FLAGS ${RAJA_NVCC_FLAGS}; -Xcompiler -coverage; -Xlinker -coverage)
      set(CMAKE_EXE_LINKER_FLAGS "-coverage ${CMAKE_EXE_LINKER_FLAGS}")
    else()
      message(WARNING "Code coverage specified but not enabled -- GCC was not detected")
    endif()
  endif()
  set (CUDA_NVCC_FLAGS ${RAJA_NVCC_FLAGS})
endif()

set(RAJA_RANGE_ALIGN 4 CACHE INT "")
set(RAJA_RANGE_MIN_LENGTH 32 CACHE INT "")
set(RAJA_DATA_ALIGN 64 CACHE INT "")
set(RAJA_COHERENCE_BLOCK_SIZE 64 CACHE INT "")

include(CheckFunctionExists)
check_function_exists(posix_memalign HAVE_POSIX_MEMALIGN)
if(${HAVE_POSIX_MEMALIGN})
  add_definitions(-DHAVE_POSIX_MEMALIGN)
endif(${HAVE_POSIX_MEMALIGN})
