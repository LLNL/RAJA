###############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if (ENABLE_CUDA OR ENABLE_HIP)
  raja_add_test(
    NAME test-chai-nested
    SOURCES chai-nested.cpp)
endif(ENABLE_CUDA OR ENABLE_HIP)

raja_add_test(
  NAME test-chai-policy
  SOURCES chai-policy-tests.cpp)

raja_add_test(
  NAME test-chai
  SOURCES chai-tests.cpp)
