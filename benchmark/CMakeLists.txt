###############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
###############################################################################

if (ENABLE_CUDA)
  raja_add_benchmark(
    NAME benchmark-host-device-lambda
    SOURCES host-device-lambda-benchmark.cpp)
endif()
