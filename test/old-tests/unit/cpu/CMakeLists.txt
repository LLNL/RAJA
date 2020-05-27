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
  NAME test-synchronize
  SOURCES test-synchronize.cpp)

if(NOT RAJA_ENABLE_TARGET_OPENMP)
  raja_add_test(
    NAME test-reductions
    SOURCES test-reductions.cpp)
endif()
