###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-synchronize
  SOURCES test-synchronize.cpp)

if(NOT RAJA_ENABLE_TARGET_OPENMP)
  raja_add_test(
    NAME test-reductions
    SOURCES test-reductions.cpp)
endif()
