###############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

add_subdirectory(atomic)

raja_add_test(
  NAME test-forall-functional
  SOURCES test-forall.cpp)

