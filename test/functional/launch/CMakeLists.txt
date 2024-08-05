###############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

list(APPEND LAUNCH_BACKENDS Sequential)

if(RAJA_ENABLE_OPENMP)
  list(APPEND LAUNCH_BACKENDS OpenMP)
endif()

if(RAJA_ENABLE_CUDA)
  list(APPEND LAUNCH_BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND LAUNCH_BACKENDS Hip)
endif()

if(RAJA_ENABLE_SYCL)
  list(APPEND LAUNCH_BACKENDS Sycl)
endif()

add_subdirectory(run-time-switch)

#Adapted from forall test
add_subdirectory(multi-reduce-nested)

add_subdirectory(reduce-basic)

add_subdirectory(reduce-params)

add_subdirectory(segment)

add_subdirectory(shared_mem)

add_subdirectory(nested_loop)

add_subdirectory(nested_direct)

add_subdirectory(tile_icount_direct)

add_subdirectory(tile_icount_loop)

add_subdirectory(nested_tile_direct)

add_subdirectory(nested_tile_loop)

unset( LAUNCH_BACKENDS )
