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

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0" CACHE STRING "")

if (CMAKE_CXX_COMPILER_ID MATCHES Clang)
  set(RAJA_COMPILER "RAJA_COMPILER_CLANG")
elseif (CMAKE_CXX_COMPILER_ID MATCHES GNU)
  set(RAJA_COMPILER "RAJA_COMPILER_GNU")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
    message(FATAL_ERROR "RAJA requires GCC 4.9 or greater!")
  endif ()
elseif (CMAKE_CXX_COMPILER_ID MATCHES Intel)
  set(RAJA_COMPILER "RAJA_COMPILER_ICC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(CMAKE_CXX_COMPILER_ID MATCHES XL)
  set(RAJA_COMPILER "RAJA_COMPILER_XLC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
  set(RAJA_COMPILER "RAJA_COMPILER_${CMAKE_CXX_COMPILER_ID}")
endif()

if (RAJA_ENABLE_CUDA)
  if(CMAKE_BUILD_TYPE MATCHES Release)
    set(RAJA_NVCC_FLAGS -O2; -restrict; -arch compute_35; -std c++11; --expt-extended-lambda; -x cu; -ccbin; ${CMAKE_CXX_COMPILER})
  elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
    set(RAJA_NVCC_FLAGS -g; -G; -O2; -restrict; -arch compute_35; -std c++11; --expt-extended-lambda; -x cu; -ccbin ${CMAKE_CXX_COMPILER})
  elseif(CMAKE_BUILD_TYPE MATCHES Debug)
    set(RAJA_NVCC_FLAGS -g; -G; -O0; -restrict; -arch compute_35; -std c++11; --expt-extended-lambda; -x cu; -ccbin ${CMAKE_CXX_COMPILER})
  endif()
endif()

set(RAJA_RANGE_ALIGN 4 CACHE INT "")
set(RAJA_RANGE_MIN_LENGTH 32 CACHE INT "")
set(RAJA_DATA_ALIGN 64 CACHE INT "")
set(RAJA_COHERENCE_BLOCK_SIZE 64 CACHE INT "")
