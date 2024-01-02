###############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-indexset
  SOURCES test-indexset.cpp)

raja_add_test(
  NAME test-indexvalue
  SOURCES test-indexvalue.cpp)

raja_add_test(
  NAME test-listsegment
  SOURCES test-listsegment.cpp)

raja_add_test(
  NAME test-rangesegment
  SOURCES test-rangesegment.cpp)

raja_add_test(
  NAME test-rangestridesegment
  SOURCES test-rangestridesegment.cpp)
