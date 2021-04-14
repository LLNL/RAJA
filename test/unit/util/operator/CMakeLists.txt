###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-operators-basic
  SOURCES test-operators-basic.cpp)

raja_add_test(
  NAME test-operators-logical
  SOURCES test-operators-logical.cpp)

#raja_add_test(
#  NAME test-operators-min-max
#  SOURCES test-operators-min-max.cpp)

raja_add_test(
  NAME test-operators-equivalence
  SOURCES test-operators-equivalence.cpp)

raja_add_test(
  NAME test-operators-bitwise-modulus
  SOURCES test-operators-bitwise-modulus.cpp)

raja_add_test(
  NAME test-operators-identity-project
  SOURCES test-operators-identity-project.cpp)

