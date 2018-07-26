###############################################################################
#
# Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-689114
#
# All rights reserved.
#
# This file is part of RAJA.
#
# For details about use and distribution, please read RAJA/LICENSE.
#
###############################################################################

raja_add_test(
  NAME test-cuda-nested
  SOURCES test-nested.cpp)

if (NOT ENABLE_CLANG_CUDA)
  raja_add_test(
    NAME test-cuda-scan
    SOURCES test-scan.cpp)
endif()

raja_add_test(
  NAME test-cuda-reduce-sum
  SOURCES test-reduce-sum.cpp)

raja_add_test(
  NAME test-cuda-reduce-min
  SOURCES test-reduce-min.cpp)

raja_add_test(
  NAME test-cuda-reduce-loc
  SOURCES test-reduce-loc.cpp)

raja_add_test(
  NAME test-cuda-reduce-max
  SOURCES test-reduce-max.cpp)

raja_add_test(
  NAME test-cuda-forall
  SOURCES test-forall.cpp)

raja_add_test(
  NAME test-cuda-nested-strided
  SOURCES test-nested-strided.cpp)

raja_add_test(
  NAME test-cuda-forall-view
  SOURCES test-forall-view.cpp)

raja_add_test(
  NAME test-cuda-synchronize
  SOURCES test-synchronize.cpp)
