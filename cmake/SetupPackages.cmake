###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
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