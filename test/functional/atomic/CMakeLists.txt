###############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-atomic-forall-basic
  SOURCES test-atomic-forall-basic.cpp)

raja_add_test(
  NAME test-atomic-forall-view
  SOURCES test-atomic-forall-view.cpp)

raja_add_test(
  NAME test-atomic-ref-forall-math
  SOURCES test-atomic-ref-forall-math.cpp)

raja_add_test(
  NAME test-atomic-ref-forall-math-auto
  SOURCES test-atomic-ref-forall-math-auto.cpp)

raja_add_test(
  NAME test-atomic-ref-forall-other
  SOURCES test-atomic-ref-forall-other.cpp)

raja_add_test(
  NAME test-atomic-ref-forall-other-auto
  SOURCES test-atomic-ref-forall-other-auto.cpp)

