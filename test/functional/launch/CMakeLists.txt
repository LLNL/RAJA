###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
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

add_subdirectory(run-time-switch)

add_subdirectory(segment)

#add_subdirectory(shared_mem)

unset( LAUNCH_BACKENDS )
