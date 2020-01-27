###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(CMAKE_BUILD_TYPE Release CACHE BOOL "")

set(ENABLE_HIP ON CACHE BOOL "")
set(ENABLE_OPENMP OFF CACHE BOOL "")
set(ENABLE_CUDA Off CACHE BOOL "")

set(HIP_ROOT_DIR "/opt/rocm/hip" CACHE PATH "HIP ROOT directory path")

set(CMAKE_CXX_COMPILER "g++" CACHE PATH "")
set(CMAKE_C_COMPILER "gcc" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE "-O2" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g" CACHE STRING "")

# set(HIP_COMMON_OPT_FLAGS  "--amdgpu-target=gfx900")
set(HIP_COMMON_OPT_FLAGS )
set(HIP_COMMON_DEBUG_FLAGS)
set(HOST_OPT_FLAGS)

if (ENABLE_OPENMP)
	set(HIP_COMMON_OPT_FLAGS "-fopenmp ${HIP_COMMON_OPT_FLAGS}")
endif()

if(CMAKE_BUILD_TYPE MATCHES Release)
  set(RAJA_HIPCC_FLAGS -O2; ${HIP_COMMON_OPT_FLAGS}; ${HOST_OPT_FLAGS} CACHE LIST "")
elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  set(RAJA_HIPCC_FLAGS -g; -G; -O2; ${HIP_COMMON_OPT_FLAGS}; ${HOST_OPT_FLAGS} CACHE LIST "")
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  set(RAJA_HIPCC_FLAGS -g; -G; -O0; ${HIP_COMMON_DEBUG_FLAGS}; CACHE LIST "")
endif()

set(RAJA_RANGE_ALIGN 4 CACHE INT "")
set(RAJA_RANGE_MIN_LENGTH 32 CACHE INT "")
set(RAJA_DATA_ALIGN 64 CACHE INT "")
set(RAJA_COHERENCE_BLOCK_SIZE 64 CACHE INT "")

set(RAJA_HOST_CONFIG_LOADED On CACHE Bool "")
