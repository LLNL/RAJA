###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

list(APPEND GRAPH_BACKENDS Sequential)

list(APPEND GRAPH_FORALL_BACKENDS Sequential)

if(RAJA_ENABLE_OPENMP)
  list(APPEND GRAPH_FORALL_BACKENDS OpenMP)
endif()

if(RAJA_ENABLE_TBB)
  list(APPEND GRAPH_FORALL_BACKENDS TBB)
endif()

if(RAJA_ENABLE_CUDA)
  list(APPEND GRAPH_FORALL_BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND GRAPH_FORALL_BACKENDS Hip)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  list(APPEND GRAPH_FORALL_BACKENDS OpenMPTarget)
endif()

add_subdirectory(ForallNode)

add_subdirectory(MixedNodes)

unset( GRAPH_FORALL_BACKENDS )
unset( GRAPH_BACKENDS )
