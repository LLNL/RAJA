##############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_executable(
  NAME atomic-histogram
  SOURCES atomic-histogram.cpp)
raja_add_executable(
  NAME atomic-histogram_solution
  SOURCES atomic-histogram_solution.cpp)

add_subdirectory(tutorial_halfday)
