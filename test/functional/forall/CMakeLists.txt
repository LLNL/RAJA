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

raja_add_test_subdirectory(indexset)
raja_add_test_subdirectory(indexset-view)

raja_add_test_subdirectory(segment)
raja_add_test_subdirectory(segment-view)

raja_add_test_subdirectory(reduce-basic)

unset( FORALL_BACKENDS ) 


#
# Note: Forall atomic tests use their own backend list since atomic 
#       constructs are defined for only some of the RAJA back-ends.
#
list(APPEND FORALL_ATOMIC_BACKENDS Sequential)

if(RAJA_ENABLE_OPENMP)
  list(APPEND FORALL_ATOMIC_BACKENDS OpenMP)
endif()

if(RAJA_ENABLE_TBB)
# TBB atomics are not available in RAJA currently
#  list(APPEND FORALL_BACKENDS TBB)
endif()

if(RAJA_ENABLE_CUDA)
  list(APPEND FORALL_ATOMIC_BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND FORALL_ATOMIC_BACKENDS Hip)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  list(APPEND FORALL_ATOMIC_BACKENDS OpenMPTarget)
endif()

raja_add_test_subdirectory(atomic-basic)
raja_add_test_subdirectory(atomic-view)
raja_add_test_subdirectory(atomic-ref)

unset( FORALL_ATOMIC_BACKENDS )

#
# Note: Forall region tests define their backend list in the region
#       test directory since region constructs are defined for only 
#       some of the RAJA back-ends.
#
raja_add_test_subdirectory(region)
