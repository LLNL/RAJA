###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-forall-reduce-sanity-seq
  SOURCES test-forall-reduce-sanity-seq.cpp)

raja_add_test(
  NAME test-forall-reduce-sanity-bit-seq
  SOURCES test-forall-reduce-sanity-bit-seq.cpp)

if(RAJA_ENABLE_OPENMP)
  raja_add_test(
    NAME test-forall-reduce-sanity-openmp
    SOURCES test-forall-reduce-sanity-openmp.cpp)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  raja_add_test(
    NAME test-forall-reduce-sanity-openmp-target
    SOURCES test-forall-reduce-sanity-openmp-target.cpp)
endif()
 
if(RAJA_ENABLE_TBB)
  raja_add_test(
    NAME test-forall-reduce-sanity-tbb
    SOURCES test-forall-reduce-sanity-tbb.cpp)
endif()

if(RAJA_ENABLE_CUDA)
  raja_add_test(
    NAME test-forall-reduce-sanity-cuda
    SOURCES test-forall-reduce-sanity-cuda.cpp)

  raja_add_test(
    NAME test-forall-reduce-sanity-bit-cuda
    SOURCES test-forall-reduce-sanity-bit-cuda.cpp)
endif()
 
if(RAJA_ENABLE_HIP)
  raja_add_test(
    NAME test-forall-reduce-sanity-hip
    SOURCES test-forall-reduce-sanity-hip.cpp)
endif()
