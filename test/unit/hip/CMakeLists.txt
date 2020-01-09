###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-hip-scan
  SOURCES test-scan.cpp)

raja_add_test(
  NAME test-hip-reduce-sum
  SOURCES test-reduce-sum.cpp)

raja_add_test(
  NAME test-hip-reduce-min
  SOURCES test-reduce-min.cpp)

raja_add_test(
  NAME test-hip-reduce-loc
  SOURCES test-reduce-loc.cpp)

raja_add_test(
  NAME test-hip-reduce-tupleloc
  SOURCES test-reduce-tupleloc.cpp)

raja_add_test(
  NAME test-hip-reduce-randloc
  SOURCES test-reduce-randloc.cpp)

raja_add_test(
  NAME test-hip-reduce-max
  SOURCES test-reduce-max.cpp)

raja_add_test(
  NAME test-hip-forall
  SOURCES test-forall.cpp)

raja_add_test(
  NAME test-hip-forall-view
  SOURCES test-forall-view.cpp)

raja_add_test(
  NAME test-hip-synchronize
  SOURCES test-synchronize.cpp)
