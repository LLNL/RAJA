###############################################################################
#
# Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-689114
#
# All rights reserved.
#
# This file is part of RAJA.
#
# For details about use and distribution, please read RAJA/LICENSE.
#
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
  SOURCES test-indexsets.cpp
  DEPENDS_ON bis)

raja_add_test(
  NAME test-nested
  SOURCES test-nested.cpp)

raja_add_test(
  NAME test-nested-reduce
  SOURCES test-nested-reduce.cpp)

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
