##############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_executable(
  NAME vector-addition
  SOURCES vector-addition.cpp)
raja_add_executable(
  NAME vector-addition_solution
  SOURCES vector-addition_solution.cpp)

raja_add_executable(
  NAME vertexsum-indexset
  SOURCES vertexsum-indexset.cpp)
raja_add_executable(
  NAME vertexsum-indexse_solution
  SOURCES vertexsum-indexse_solution.cpp)

raja_add_executable(
  NAME atomic-histogram
  SOURCES atomic-histogram.cpp)
raja_add_executable(
  NAME atomic-histogram_solution
  SOURCES atomic-histogram_solution.cpp)

add_subdirectory(tutorial_halfday)
