###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-kernel-region-seq
  SOURCES test-kernel-region-seq.cpp)

if(RAJA_ENABLE_OPENMP)
  raja_add_test(
    NAME test-kernel-region-openmp
    SOURCES test-kernel-region-openmp.cpp)
  raja_add_test(
    NAME test-kernel-region-sync-openmp
    SOURCES test-kernel-region-sync-openmp.cpp)
endif()
