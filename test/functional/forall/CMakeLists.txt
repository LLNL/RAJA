###############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-forall-seq
  SOURCES test-forall-seq.cpp)

raja_add_test(
  NAME test-forall-openmp
  SOURCES test-forall-openmp.cpp)

raja_add_test(
  NAME test-forall-target-openmp
  SOURCES test-forall-target-openmp.cpp)

raja_add_test(
  NAME test-forall-tbb
  SOURCES test-forall-tbb.cpp)

raja_add_test(
  NAME test-forall-cuda
  SOURCES test-forall-cuda.cpp)

raja_add_test(
  NAME test-forall-hip
  SOURCES test-forall-hip.cpp)
