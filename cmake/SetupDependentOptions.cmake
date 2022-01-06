###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and other RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
################################################################################

##
## Here are the CMake dependent options in RAJA.
##


cmake_dependent_option(RAJA_ENABLE_OPENMP "Build with OpenMP support" On "ENABLE_OPENMP" Off)
cmake_dependent_option(RAJA_ENABLE_CUDA "Build with CUDA support" On "ENABLE_CUDA" Off)
cmake_dependent_option(RAJA_ENABLE_HIP "Build with HIP support" On "ENABLE_HIP" Off)
cmake_dependent_option(RAJA_ENABLE_CLANG_CUDA "Build with Clang CUDA support" On "ENABLE_CLANG_CUDA" Off)

cmake_dependent_option(RAJA_ENABLE_COVERAGE "Enable coverage (only supported with GCC)" On "ENABLE_COVERAGE" Off)
cmake_dependent_option(RAJA_ENABLE_TESTS "Build tests" On "ENABLE_TESTS" Off)
cmake_dependent_option(RAJA_ENABLE_EXAMPLES "Build simple examples" On "ENABLE_EXAMPLES" off)
cmake_dependent_option(RAJA_ENABLE_BENCHMARKS "Build benchmarks" On "ENABLE_BENCHMARKS" Off)
