###############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

list(APPEND DYNAMIC_BACKENDS Sequential)

if(RAJA_ENABLE_OPENMP)
  list(APPEND DYNAMIC_BACKENDS OpenMP)
endif()

if(RAJA_ENABLE_CUDA)
  list(APPEND DYNAMIC_BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND DYNAMIC_BACKENDS Hip)
endif()

add_subdirectory(segment)

add_subdirectory(resource-segment)

unset( DYNAMIC_BACKENDS )
