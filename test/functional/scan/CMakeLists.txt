###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-scan-inclusive-seq
  SOURCES test-scan-inclusive-seq.cpp)

raja_add_test(
  NAME test-scan-exclusive-seq
  SOURCES test-scan-exclusive-seq.cpp)

if(RAJA_ENABLE_OPENMP)
raja_add_test(
  NAME test-scan-inclusive-openmp
  SOURCES test-scan-inclusive-openmp.cpp)
raja_add_test(
  NAME test-scan-exclusive-openmp
  SOURCES test-scan-exclusive-openmp.cpp)
endif()

if(RAJA_ENABLE_TBB)
raja_add_test(
  NAME test-scan-inclusive-tbb
  SOURCES test-scan-inclusive-tbb.cpp)
raja_add_test(
  NAME test-scan-exclusive-tbb
  SOURCES test-scan-exclusive-tbb.cpp)
endif()
