###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if(ENABLE_TARGET_OPENMP)
  raja_add_test(
    NAME test-omp-target-nested-reduce
    SOURCES test-nested-reduce.cpp)
  raja_add_test(
    NAME test-omp-target-reductions
    SOURCES test-reductions.cpp)
  raja_add_test(
    NAME test-omp-target-reduce-tupleminloc
    SOURCES test-reduce-tupleminloc.cpp)
  raja_add_test(
    NAME test-omp-target-reduce-tuplemaxloc
    SOURCES test-reduce-tuplemaxloc.cpp)
  raja_add_test(
    NAME test-omp-target-kernel
    SOURCES kernel.cpp)
endif(ENABLE_TARGET_OPENMP)
