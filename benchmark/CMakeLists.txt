###############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if (RAJA_ENABLE_CUDA)
  raja_add_benchmark(
    NAME benchmark-host-device-lambda
    SOURCES host-device-lambda-benchmark.cpp)
endif()

if (RAJA_ENABLE_DESUL)
raja_add_benchmark(
  NAME benchmark-atomic
  SOURCES benchmark-atomic.cpp)
endif()

#TODO: Fix build issues
#raja_add_benchmark(
#  NAME ltimes
#  SOURCES ltimes.cpp)

raja_add_benchmark(
  NAME raja_view_blur
  SOURCES raja_view_blur.cpp)
