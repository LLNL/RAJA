###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and other RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
################################################################################

set(RAJA_ENABLE_WARNINGS_AS_ERRORS Off CACHE BOOL "")
set(ENABLE_GTEST_DEATH_TESTS On CACHE BOOL "Enable tests asserting failure.")

option(RAJA_ENABLE_NV_TOOLS_EXT "Build with NV_TOOLS_EXT support" Off)
option(RAJA_ENABLE_ROCTX "Build with ENABLE_ROCTX support" Off)

option(RAJA_ENABLE_TBB "Build TBB support" Off)
option(RAJA_ENABLE_TARGET_OPENMP "Build OpenMP on target device support" Off)
option(RAJA_ENABLE_SYCL "Build SYCL support" Off)

option(RAJA_ENABLE_EXPTVECTOR "Build experimental vectorization support" Off)

option(RAJA_ENABLE_REPRODUCERS "Build issue reproducers" Off)

option(RAJA_ENABLE_EXERCISES "Build exercises " On)
option(RAJA_ENABLE_WARNINGS "Enable warnings as errors for CI" Off)
option(RAJA_ENABLE_DOCUMENTATION "Build RAJA documentation" Off)
option(RAJA_ENABLE_FORCEINLINE_RECURSIVE "Enable Forceinline recursive (only supported with Intel compilers)" On)

option(RAJA_DEPRECATED_TESTS "Test deprecated features" Off)
option(RAJA_ENABLE_BOUNDS_CHECK "Enable bounds checking in RAJA::Views/Layouts" Off)
option(RAJA_TEST_EXHAUSTIVE "Build RAJA exhaustive tests" Off)
option(RAJA_TEST_OPENMP_TARGET_SUBSET "Build subset of RAJA OpenMP target tests when it is enabled" On)
option(RAJA_ENABLE_RUNTIME_PLUGINS "Enable support for loading plugins at runtime" Off)
option(RAJA_ALLOW_INCONSISTENT_OPTIONS "Enable inconsistent values for ENABLE_X and RAJA_ENABLE_X options" Off)

option(RAJA_ENABLE_DESUL_ATOMICS "Enable support of desul atomics" Off)
set(DESUL_ENABLE_TESTS Off CACHE BOOL "")

set(TEST_DRIVER "" CACHE STRING "driver used to wrap test commands")

set(BLT_EXPORT_THIRDPARTY ON CACHE BOOL "")
