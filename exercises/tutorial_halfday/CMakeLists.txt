###############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_executable(
  NAME ex1_vector-addition_solution
  SOURCES ex1_vector-addition_solution.cpp)

raja_add_executable(
  NAME ex2_approx-pi_solution
  SOURCES ex2_approx-pi_solution.cpp)

raja_add_executable(
  NAME ex3_colored-indexset_solution
  SOURCES ex3_colored-indexset_solution.cpp)

raja_add_executable(
  NAME ex4_atomic-histogram_solution
  SOURCES ex4_atomic-histogram_solution.cpp)

raja_add_executable(
  NAME ex5_line-of-sight_solution
  SOURCES ex5_line-of-sight_solution.cpp)

raja_add_executable(
  NAME ex6_stencil-offset-layout_solution
  SOURCES ex6_stencil-offset-layout_solution.cpp)
