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

if(RAJA_ENABLE_CUDA)

raja_add_test(
  NAME test-scan-inclusive-cuda
  SOURCES test-scan-inclusive-cuda.cpp)
raja_add_test(
  NAME test-scan-exclusive-cuda
  SOURCES test-scan-exclusive-cuda.cpp)

endif()

if(RAJA_ENABLE_HIP)

raja_add_test(
  NAME test-scan-inclusive-hip
  SOURCES test-scan-inclusive-hip.cpp)
raja_add_test(
  NAME test-scan-exclusive-hip
  SOURCES test-scan-exclusive-hip.cpp)

endif()
