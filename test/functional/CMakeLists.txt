###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

add_subdirectory(forall)

add_subdirectory(dynamic_forall)

add_subdirectory(indexset-build)

add_subdirectory(kernel)

add_subdirectory(scan)

add_subdirectory(workgroup)

add_subdirectory(teams)

if (RAJA_ENABLE_VECTORIZATION)
  add_subdirectory(tensor)
endif()

add_subdirectory(util)
