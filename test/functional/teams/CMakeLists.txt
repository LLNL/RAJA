###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

list(APPEND TEAMS_BACKENDS Sequential)

if(RAJA_ENABLE_OPENMP)
  list(APPEND TEAMS_BACKENDS OpenMP)
endif()

if(RAJA_ENABLE_CUDA)
  list(APPEND TEAMS_BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND TEAMS_BACKENDS Hip)
endif()

add_subdirectory(run-time-switch)

unset( TEAMS_BACKENDS )
