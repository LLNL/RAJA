###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and other RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if (ENABLE_OPENMP)
  if(OPENMP_FOUND)
    list(APPEND RAJA_EXTRA_NVCC_FLAGS -Xcompiler ${OpenMP_CXX_FLAGS})
    message(STATUS "OpenMP Enabled")
  else()
    message(WARNING "OpenMP NOT FOUND")
    set(ENABLE_OPENMP Off)
  endif()
endif()

if (ENABLE_TBB)
  find_package(TBB)
  if(TBB_FOUND)
    blt_register_library(
      NAME tbb
      INCLUDES ${TBB_INCLUDE_DIRS}
      LIBRARIES ${TBB_LIBRARIES})
    message(STATUS "TBB Enabled")
  else()
    message(WARNING "TBB NOT FOUND")
    set(ENABLE_TBB Off)
  endif()
endif ()

if (ENABLE_CUDA OR ENABLE_EXTERNAL_CUB)
  find_package(CUB)
  if (CUB_FOUND)
    set(ENABLE_EXTERNAL_CUB On)
    blt_register_library(
      NAME cub
      INCLUDES ${CUB_INCLUDE_DIRS})
  elseif(ENABLE_EXTERNAL_CUB)
    message(FATAL_ERROR "External CUB not found, CUB_DIR=${CUB_DIR}.")
  else()
    message(STATUS "Using RAJA CUB submodule.")
  endif()
endif ()

if (ENABLE_HIP OR ENABLE_EXTERNAL_ROCPRIM)
  find_package(RocPRIM)
  if (ROCPRIM_FOUND)
    set(ENABLE_EXTERNAL_ROCPRIM On)
    blt_register_library(
      NAME rocPRIM
      INCLUDES ${ROCPRIM_INCLUDE_DIRS})
  elseif (ENABLE_EXTERNAL_ROCPRIM)
      message(FATAL_ERROR "External rocPRIM not found, ROCPRIM_DIR=${ROCPRIM_DIR}.")
  else()
    message(STATUS "Using RAJA rocPRIM submodule.")
  endif()
endif ()
