###############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

include_directories(include)

add_subdirectory(integration)

add_subdirectory(functional)

add_subdirectory(unit)

add_subdirectory(old-tests)

add_subdirectory(install)

configure_file(${CMAKE_SOURCE_DIR}/test/CTestCustom.cmake ${CMAKE_BINARY_DIR})
