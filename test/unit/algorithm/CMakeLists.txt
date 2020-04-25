###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-sort-seq
  SOURCES test-sort-seq.cpp)

raja_add_test(
  NAME test-stable-sort-seq
  SOURCES test-stable-sort-seq.cpp)

raja_add_test(
  NAME test-insertion-sort-cpu
  SOURCES test-insertion-sort-cpu.cpp)

raja_add_test(
  NAME test-shell-sort-cpu
  SOURCES test-shell-sort-cpu.cpp)

raja_add_test(
  NAME test-heap-sort-cpu
  SOURCES test-heap-sort-cpu.cpp)

raja_add_test(
  NAME test-intro-sort-cpu
  SOURCES test-intro-sort-cpu.cpp)

raja_add_test(
  NAME test-merge-sort-cpu
  SOURCES test-merge-sort-cpu.cpp)

if(RAJA_ENABLE_OPENMP)
  raja_add_test(
    NAME test-sort-openmp
    SOURCES test-sort-openmp.cpp)

  raja_add_test(
    NAME test-stable-sort-openmp
    SOURCES test-stable-sort-openmp.cpp)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  # reserved for futute compatability
endif()

if(RAJA_ENABLE_TBB)
  raja_add_test(
    NAME test-sort-tbb
    SOURCES test-sort-tbb.cpp)

  raja_add_test(
    NAME test-stable-sort-tbb
    SOURCES test-stable-sort-tbb.cpp)
endif()

if(RAJA_ENABLE_CUDA)
  raja_add_test(
    NAME test-sort-cuda
    SOURCES test-sort-cuda.cpp)

  raja_add_test(
    NAME test-stable-sort-cuda
    SOURCES test-stable-sort-cuda.cpp)

  raja_add_test(
    NAME test-insertion-sort-cuda
    SOURCES test-insertion-sort-cuda.cpp)

  raja_add_test(
    NAME test-shell-sort-cuda
    SOURCES test-shell-sort-cuda.cpp)

  raja_add_test(
    NAME test-heap-sort-cuda
    SOURCES test-heap-sort-cuda.cpp)

  raja_add_test(
    NAME test-intro-sort-cuda
    SOURCES test-intro-sort-cuda.cpp)

  raja_add_test(
    NAME test-merge-sort-cuda
    SOURCES test-merge-sort-cuda.cpp)
endif()

if(RAJA_ENABLE_HIP)
  raja_add_test(
    NAME test-sort-hip
    SOURCES test-sort-hip.cpp)

  raja_add_test(
    NAME test-stable-sort-hip
    SOURCES test-stable-sort-hip.cpp)

  raja_add_test(
    NAME test-insertion-sort-hip
    SOURCES test-insertion-sort-hip.cpp)

  raja_add_test(
    NAME test-shell-sort-hip
    SOURCES test-shell-sort-hip.cpp)

  raja_add_test(
    NAME test-heap-sort-hip
    SOURCES test-heap-sort-hip.cpp)

  raja_add_test(
    NAME test-intro-sort-hip
    SOURCES test-intro-sort-hip.cpp)

  raja_add_test(
    NAME test-merge-sort-hip
    SOURCES test-merge-sort-hip.cpp)
endif()

