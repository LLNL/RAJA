###############################################################################
# Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
#    
# Produced at the Lawrence Livermore National Laboratory
#    
# LLNL-CODE-689114
# 
# All rights reserved.
#  
# This file is part of RAJA.
#
# For details about use and distribution, please read RAJA/LICENSE.
#
###############################################################################

if (ENABLE_OPENMP)
  if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    list(APPEND RAJA_EXTRA_NVCC_FLAGS -Xcompiler ${OpenMP_CXX_FLAGS})
    message(STATUS "OpenMP Enabled")
  else()
    message(WARNING "OpenMP NOT FOUND")
    set(ENABLE_OPENMP Off)
  endif()
endif()

if (ENABLE_CUDA)
  if (ENABLE_CUB)
    find_package(CUB)
    if (CUB_FOUND)
      blt_register_library(
        NAME cub
        INCLUDES ${CUB_INCLUDE_DIRS})
    else()
      message(WARNING "Using deprecated Thrust backend for CUDA scans.\n
  Please set CUB_DIR for better scan performance.")
      set(ENABLE_CUB Off)
    endif()
  endif()
endif ()

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

if (ENABLE_CHAI)
  message(STATUS "CHAI enabled")
  find_package(chai)
  include_directories(${CHAI_INCLUDE_DIRS})
endif()
