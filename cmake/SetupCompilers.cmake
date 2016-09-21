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

  set(RAJA_ALIGNED_ATTR "alignas(N)")
  set(RAJA_INLINE "inline  __attribute__((always_inline))")
  set(RAJA_SIMD "")
  set(RAJA_ALIGN_DATA "")

elseif (CMAKE_CXX_COMPILER_ID MATCHES GNU)
  set(RAJA_COMPILER "RAJA_COMPILER_GNU")

  set(RAJA_ALIGNED_ATTR "__attribute__((aligned(N)))")
  set(RAJA_INLINE "inline  __attribute__((always_inline))")
  set(RAJA_ALIGN_DATA "__builtin_assume_aligned(d, DATA_ALIGN)")
  set(RAJA_SIMD "")

elseif (CMAKE_CXX_COMPILER_ID MATCHES Intel)
  set(RAJA_COMPILER "RAJA_COMPILER_ICC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

  set(RAJA_ALIGNED_ATTR "alignas(N)")
  set(RAJA_INLINE "inline  __attribute__((always_inline))")
  set(RAJA_SIMD "")
  set(RAJA_ALIGN_DATA "")

else()
  set(RAJA_COMPILER "RAJA_COMPILER_${CMAKE_CXX_COMPILER_ID}")

  set(RAJA_ALIGNED_ATTR "alignas(N)")
  set(RAJA_INLINE "inline")
  set(RAJA_ALIGN_DATA "")
  set(RAJA_SIMD "")
endif()

if (RAJA_ENABLE_CUDA)
  set(RAJA_ALIGN_DATA "")

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

# INTEL
#if __ICC < 1300  // use alignment intrinsic
#define RAJA_ALIGN_DATA(d) __assume_aligned(d, DATA_ALIGN)
#else
#define RAJA_ALIGN_DATA(d)  // TODO: Define this...
#endif
#endif
#define RAJA_SIMD  // TODO: Define this...

# XLC 12
#define RAJA_INLINE inline  __attribute__((always_inline))
#define RAJA_ALIGN_DATA(d) __alignx(DATA_ALIGN, d)
#define RAJA_SIMD  _Pragma("simd_level(10)")
#define RAJA_SIMD   // TODO: Define this... 
