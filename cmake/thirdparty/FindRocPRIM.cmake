###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and other RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

include (FindPackageHandleStandardArgs)

find_path(ROCPRIM_INCLUDE_DIRS
  NAMES rocprim/rocprim.hpp
  HINTS
    ${ROCPRIM_DIR}/
    ${ROCPRIM_DIR}/include
    ${ROCPRIM_DIR}/rocprim/include
    ${HIP_ROOT_DIR}/../rocprim
    ${HIP_ROOT_DIR}/../rocprim/include
    ${HIP_ROOT_DIR}/../include)

find_package_handle_standard_args(
  ROCPRIM
  DEFAULT_MSG
  ROCPRIM_INCLUDE_DIRS)

if (ROCPRIM_INCLUDE_DIRS)
  set(ROCPRIM_FOUND True)
else ()
  set(ROCPRIM_FOUND False)
endif()
