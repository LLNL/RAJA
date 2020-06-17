###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-forall-reducesum-sanity-seq
  SOURCES test-forall-reducesum-sanity-seq.cpp)

raja_add_test(
  NAME test-forall-reducemin-sanity-seq
  SOURCES test-forall-reducemin-sanity-seq.cpp)

raja_add_test(
  NAME test-forall-reducemax-sanity-seq
  SOURCES test-forall-reducemax-sanity-seq.cpp)

raja_add_test(
  NAME test-forall-reduceminloc-sanity-seq
  SOURCES test-forall-reduceminloc-sanity-seq.cpp)

raja_add_test(
  NAME test-forall-reducemaxloc-sanity-seq
  SOURCES test-forall-reducemaxloc-sanity-seq.cpp)

if(RAJA_ENABLE_OPENMP)
  raja_add_test(
    NAME test-forall-reducesum-sanity-openmp
    SOURCES test-forall-reducesum-sanity-openmp.cpp)

  raja_add_test(
    NAME test-forall-reducemin-sanity-openmp
    SOURCES test-forall-reducemin-sanity-openmp.cpp)

  raja_add_test(
    NAME test-forall-reducemax-sanity-openmp
    SOURCES test-forall-reducemax-sanity-openmp.cpp)

  raja_add_test(
    NAME test-forall-reduceminloc-sanity-openmp
    SOURCES test-forall-reduceminloc-sanity-openmp.cpp)

  raja_add_test(
    NAME test-forall-reducemaxloc-sanity-openmp
    SOURCES test-forall-reducemaxloc-sanity-openmp.cpp)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  raja_add_test(
    NAME test-forall-reducesum-sanity-openmp-target
    SOURCES test-forall-reducesum-sanity-openmp-target.cpp)

  raja_add_test(
    NAME test-forall-reducemin-sanity-openmp-target
    SOURCES test-forall-reducemin-sanity-openmp-target.cpp)

  raja_add_test(
    NAME test-forall-reducemax-sanity-openmp-target
    SOURCES test-forall-reducemax-sanity-openmp-target.cpp)

  raja_add_test(
    NAME test-forall-reduceminloc-sanity-openmp-target
    SOURCES test-forall-reduceminloc-sanity-openmp-target.cpp)

  raja_add_test(
    NAME test-forall-reducemaxloc-sanity-openmp-target
    SOURCES test-forall-reducemaxloc-sanity-openmp-target.cpp)
endif()
 
if(RAJA_ENABLE_TBB)
  raja_add_test(
    NAME test-forall-reducesum-sanity-tbb
    SOURCES test-forall-reducesum-sanity-tbb.cpp)

  raja_add_test(
    NAME test-forall-reducemin-sanity-tbb
    SOURCES test-forall-reducemin-sanity-tbb.cpp)

  raja_add_test(
    NAME test-forall-reducemax-sanity-tbb
    SOURCES test-forall-reducemax-sanity-tbb.cpp)

  raja_add_test(
    NAME test-forall-reduceminloc-sanity-tbb
    SOURCES test-forall-reduceminloc-sanity-tbb.cpp)

  raja_add_test(
    NAME test-forall-reducemaxloc-sanity-tbb
    SOURCES test-forall-reducemaxloc-sanity-tbb.cpp)
endif()

if(RAJA_ENABLE_CUDA)
  raja_add_test(
    NAME test-forall-reducesum-sanity-cuda
    SOURCES test-forall-reducesum-sanity-cuda.cpp)

  raja_add_test(
    NAME test-forall-reducemin-sanity-cuda
    SOURCES test-forall-reducemin-sanity-cuda.cpp)

  raja_add_test(
    NAME test-forall-reducemax-sanity-cuda
    SOURCES test-forall-reducemax-sanity-cuda.cpp)

  raja_add_test(
    NAME test-forall-reduceminloc-sanity-cuda
    SOURCES test-forall-reduceminloc-sanity-cuda.cpp)

  raja_add_test(
    NAME test-forall-reducemaxloc-sanity-cuda
    SOURCES test-forall-reducemaxloc-sanity-cuda.cpp)
endif()
 
if(RAJA_ENABLE_HIP)
    raja_add_test(
    NAME test-forall-reducesum-sanity-hip
    SOURCES test-forall-reducesum-sanity-hip.cpp)
    
  raja_add_test(
    NAME test-forall-reducemin-sanity-hip
    SOURCES test-forall-reducemin-sanity-hip.cpp)
  
  raja_add_test(
    NAME test-forall-reducemax-sanity-hip
    SOURCES test-forall-reducemax-sanity-hip.cpp)

  raja_add_test(
    NAME test-forall-reduceminloc-sanity-hip
    SOURCES test-forall-reduceminloc-sanity-hip.cpp)

  raja_add_test(
    NAME test-forall-reducemaxloc-sanity-hip
    SOURCES test-forall-reducemaxloc-sanity-hip.cpp)
endif()
