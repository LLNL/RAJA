###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

list(APPEND FORALL_BACKENDS Sequential)

if(RAJA_ENABLE_OPENMP)
  list(APPEND FORALL_BACKENDS OpenMP)
endif()

if(RAJA_ENABLE_TBB)
  list(APPEND FORALL_BACKENDS TBB)
endif()

if(RAJA_ENABLE_CUDA)
  list(APPEND FORALL_BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND FORALL_BACKENDS Hip)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  list(APPEND FORALL_BACKENDS OpenMPTarget)
endif()


add_subdirectory(indexset)
add_subdirectory(indexset-view)

dd_subdirectory(segment)
add_subdirectory(segment-view)

add_subdirectory(atomic-basic)
add_subdirectory(atomic-view)
add_subdirectory(atomic-ref)

add_subdirectory(reduce-sanity)

unset( FORALL_BACKENDS ) 

add_subdirectory(region)
