###############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
# and other RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

###############################################################################

# Find ROCTracer libraries/headers

find_path(ROCTX_PREFIX
  NAMES include/roctracer/roctx.h
)

find_library(ROCTX_LIBRARIES
  NAMES roctx64
  HINTS ${ROCTX_PREFIX}/lib
)

find_path(ROCTX_INCLUDE_DIRS
  NAMES roctx.h
  HINTS ${ROCTX_PREFIX}/include/roctracer
)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(ROCTX
  DEFAULT_MSG
  ROCTX_LIBRARIES
  ROCTX_INCLUDE_DIRS
)

mark_as_advanced(
  ROCTX_INCLUDE_DIRS
  ROCTX_LIBRARIES
)
