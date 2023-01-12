###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-operators-math
  SOURCES test-operators-math.cpp)

raja_add_test(
  NAME test-operators-logical
  SOURCES test-operators-logical.cpp)

raja_add_test(
  NAME test-operators-equivalence
  SOURCES test-operators-equivalence.cpp)

raja_add_test(
  NAME test-operators-bitwise-modulus
  SOURCES test-operators-bitwise-modulus.cpp)

raja_add_test(
  NAME test-operators-identity
  SOURCES test-operators-identity.cpp)

