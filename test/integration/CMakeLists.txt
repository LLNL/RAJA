###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

list(APPEND PLUGIN_BACKENDS Sequential)

if(RAJA_ENABLE_OPENMP)
  list(APPEND PLUGIN_BACKENDS OpenMP)
endif()

if(RAJA_ENABLE_TBB)
  list(APPEND PLUGIN_BACKENDS TBB)
endif()

if(RAJA_ENABLE_CUDA)
  list(APPEND PLUGIN_BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND PLUGIN_BACKENDS Hip)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  list(APPEND PLUGIN_BACKENDS OpenMPTarget)
endif()

add_subdirectory(plugin)

unset( PLUGIN_BACKENDS )
