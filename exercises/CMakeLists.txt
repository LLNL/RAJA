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

raja_add_executable(
  NAME dot-product
  SOURCES dot-product.cpp)
raja_add_executable(
  NAME dot-product_solution
  SOURCES dot-product_solution.cpp)

raja_add_executable(
  NAME reductions
  SOURCES reductions.cpp)
raja_add_executable(
  NAME reductions_solution
  SOURCES reductions_solution.cpp)

raja_add_executable(
  NAME kernelintro-execpols
  SOURCES kernelintro-execpols.cpp)

raja_add_executable(
  NAME kernelintro-execpols_solution
  SOURCES kernelintro-execpols_solution.cpp)

raja_add_executable(
  NAME launchintro-execpols
  SOURCES launchintro-execpols.cpp)

raja_add_executable(
  NAME launchintro-execpols_solution
  SOURCES launchintro-execpols_solution.cpp)

raja_add_executable(
  NAME kernelintro-nested-loop-reorder
  SOURCES kernelintro-nested-loop-reorder.cpp)
raja_add_executable(
  NAME kernelintro-nested-loop-reorder_solution
  SOURCES kernelintro-nested-loop-reorder_solution.cpp)

raja_add_executable(
  NAME kernel-matrix-transpose_solution
  SOURCES kernel-matrix-transpose_solution.cpp)

raja_add_executable(
  NAME launch-matrix-transpose_solution
  SOURCES launch-matrix-transpose_solution.cpp)

raja_add_executable(
  NAME kernel-matrix-transpose-tiled_solution
  SOURCES kernel-matrix-transpose-tiled_solution.cpp)

raja_add_executable(
  NAME kernel-matrix-transpose-local-array_solution
  SOURCES kernel-matrix-transpose-local-array_solution.cpp)

raja_add_executable(
  NAME segment-indexset-basics
  SOURCES segment-indexset-basics.cpp)
raja_add_executable(
  NAME segment-indexset-basics_solution
  SOURCES segment-indexset-basics_solution.cpp)

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
  NAME vertexsum-indexset_solution
  SOURCES vertexsum-indexset_solution.cpp)

raja_add_executable(
  NAME scan
  SOURCES scan.cpp)
raja_add_executable(
  NAME scan_solution
  SOURCES scan_solution.cpp)

raja_add_executable(
  NAME sort
  SOURCES sort.cpp)
raja_add_executable(
  NAME sort_solution
  SOURCES sort_solution.cpp)

raja_add_executable(
  NAME view-layout_solution
  SOURCES view-layout_solution.cpp)

add_subdirectory(tutorial_halfday)
