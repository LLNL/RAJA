###############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
# and other RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

include (FindPackageHandleStandardArgs)

find_path(CUB_INCLUDE_DIRS
  NAMES cub/cub.cuh
  HINTS
    ${CUB_DIR}/
    ${CUB_DIR}/include
    ${CUDA_TOOLKIT_ROOT_DIR}/include)

find_package_handle_standard_args(
  CUB
  DEFAULT_MSG
  CUB_INCLUDE_DIRS)

if (CUB_INCLUDE_DIRS)
  set(CUB_FOUND True)
else ()
  set(CUB_FOUND False)
endif()
