###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if (RAJA_ENABLE_CUDA)
  raja_add_benchmark(
    NAME benchmark-host-device-lambda
    SOURCES host-device-lambda-benchmark.cpp)
endif()

raja_add_benchmark(
  NAME ltimes
  SOURCES ltimes.cpp)
