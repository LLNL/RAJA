###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-forall-region-seq
  SOURCES test-forall-region-seq.cpp)

if(RAJA_ENABLE_OPENMP)
  raja_add_test(
    NAME test-forall-region-openmp
    SOURCES test-forall-region-openmp.cpp)
endif()
