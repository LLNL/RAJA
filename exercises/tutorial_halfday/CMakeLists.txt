###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_executable(
  NAME ex2_approx-pi
  SOURCES ex2_approx-pi.cpp)

raja_add_executable(
  NAME ex2_approx-pi_solution
  SOURCES ex2_approx-pi_solution.cpp)

raja_add_executable(
  NAME ex5_line-of-sight
  SOURCES ex5_line-of-sight.cpp)

raja_add_executable(
  NAME ex5_line-of-sight_solution
  SOURCES ex5_line-of-sight_solution.cpp)

raja_add_executable(
  NAME ex6_stencil-offset-layout
  SOURCES ex6_stencil-offset-layout.cpp)

raja_add_executable(
  NAME ex6_stencil-offset-layout_solution
  SOURCES ex6_stencil-offset-layout_solution.cpp)

raja_add_executable(
  NAME ex7_nested-loop-reorder
  SOURCES ex7_nested-loop-reorder.cpp)

raja_add_executable(
  NAME ex7_nested-loop-reorder_solution
  SOURCES ex7_nested-loop-reorder_solution.cpp)

raja_add_executable(
  NAME ex8_tiled-matrix-transpose
  SOURCES ex8_tiled-matrix-transpose.cpp)

raja_add_executable(
  NAME ex8_tiled-matrix-transpose_solution
  SOURCES ex8_tiled-matrix-transpose_solution.cpp)

raja_add_executable(
  NAME ex9_matrix-transpose-local-array
  SOURCES ex9_matrix-transpose-local-array.cpp)

raja_add_executable(
  NAME ex9_matrix-transpose-local-array_solution
  SOURCES ex9_matrix-transpose-local-array_solution.cpp)
