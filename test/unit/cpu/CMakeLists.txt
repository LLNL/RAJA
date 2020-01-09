###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

blt_add_library(
  NAME bis
  SOURCES buildIndexSet.cpp
  DEPENDS_ON RAJA ${raja_depends})

raja_add_test(
  NAME test-reduce
  SOURCES test-reduce.cpp
  DEPENDS_ON bis)

raja_add_test(
  NAME test-forall
  SOURCES test-forall.cpp
  DEPENDS_ON bis)

raja_add_test(
  NAME test-indexset
  SOURCES test-indexset.cpp
  DEPENDS_ON bis)

raja_add_test(
  NAME test-segments
  SOURCES test-segments.cpp)

raja_add_test(
  NAME test-scan
  SOURCES test-scan.cpp)

raja_add_test(
  NAME test-reductions
  SOURCES test-reductions.cpp)

raja_add_test(
  NAME test-forall-view
  SOURCES test-forall-view.cpp)

raja_add_test(
  NAME test-synchronize
  SOURCES test-synchronize.cpp)
