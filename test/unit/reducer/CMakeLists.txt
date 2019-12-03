###############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#add_subdirectory(cpu)
#add_subdirectory(cuda)
#add_subdirectory(omp-target)

raja_add_test(
  NAME test-reduce-constructors
  SOURCES test-reduce-constructors.cpp)

raja_add_test(
  NAME test-reduce-reset
  SOURCES test-reduce-reset.cpp)

