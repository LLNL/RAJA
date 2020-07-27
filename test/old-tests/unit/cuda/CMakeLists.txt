###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-cuda-reduce-sum
  SOURCES test-reduce-sum.cpp)

raja_add_test(
  NAME test-cuda-reduce-tupleloc
  SOURCES test-reduce-tupleloc.cpp)

raja_add_test(
  NAME test-cuda-reduce-randloc
  SOURCES test-reduce-randloc.cpp)

raja_add_test(
  NAME test-cuda-synchronize
  SOURCES test-synchronize.cpp)
