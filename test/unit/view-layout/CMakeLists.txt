###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-noprojection-layout
  SOURCES test-noprojection-layout.cpp)
target_include_directories(test-noprojection-layout.exe
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)

raja_add_test(
  NAME test-standard-layout
  SOURCES test-standard-layout.cpp)
target_include_directories(test-standard-layout.exe
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)

raja_add_test(
  NAME test-noprojection-typedlayout
  SOURCES test-noprojection-typedlayout.cpp)
target_include_directories(test-noprojection-typedlayout.exe
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)

raja_add_test(
  NAME test-typedlayout
  SOURCES test-typedlayout.cpp)
target_include_directories(test-typedlayout.exe
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)

raja_add_test(
  NAME test-typedview
  SOURCES test-typedview.cpp)

raja_add_test(
  NAME test-makelayout
  SOURCES test-makelayout.cpp)

raja_add_test(
  NAME test-multiview
  SOURCES test-multiview.cpp)
