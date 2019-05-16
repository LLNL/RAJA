###############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
###############################################################################

if(ENABLE_TARGET_OPENMP)
  raja_add_test(
    NAME test-omp-target-nested-reduce
    SOURCES test-nested-reduce.cpp)
  raja_add_test(
    NAME test-omp-target-reductions
    SOURCES test-reductions.cpp)
  raja_add_test(
    NAME test-omp-target-kernel
    SOURCES kernel.cpp)
endif(ENABLE_TARGET_OPENMP)
