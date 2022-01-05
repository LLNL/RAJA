###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(RAJA_COMPILER "RAJA_COMPILER_CLANG" CACHE STRING "")

set(CMAKE_CXX_FLAGS_RELEASE "-O2" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g" CACHE STRING "")

set(HIP_COMMON_OPT_FLAGS )
set(HIP_COMMON_DEBUG_FLAGS)
set(HOST_OPT_FLAGS)

if(DEFINED ROCM_ROOT_DIR)
  set(CMAKE_EXE_LINKER_FLAGS "-Wl,--disable-new-dtags -L${ROCM_ROOT_DIR}/hip/lib -L${ROCM_ROOT_DIR}/lib -L${ROCM_ROOT_DIR}/lib64 -Wl,-rpath,${ROCM_ROOT_DIR}/hip/lib:${ROCM_ROOT_DIR}/lib:${ROCM_ROOT_DIR}/lib64 -lamdhip64 -lhsakmt -lhsa-runtime64" CACHE PATH "")
endif()

if(CMAKE_BUILD_TYPE MATCHES Release)
  set(RAJA_HIPCC_FLAGS "-fPIC -O2 ${HIP_COMMON_OPT_FLAGS} ${HOST_OPT_FLAGS}" CACHE STRING "")
elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  set(RAJA_HIPCC_FLAGS "-fPIC -g -O2 ${HIP_COMMON_OPT_FLAGS} ${HOST_OPT_FLAGS}" CACHE STRING "")
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  set(RAJA_HIPCC_FLAGS "-fPIC -g -O0 ${HIP_COMMON_DEBUG_FLAGS}" CACHE STRING "")
endif()

set(RAJA_DATA_ALIGN 64 CACHE STRING "")

set(RAJA_HOST_CONFIG_LOADED On CACHE BOOL "")
