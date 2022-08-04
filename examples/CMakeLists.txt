###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_executable(
  NAME tut_teams_basic
  SOURCES tut_teams_basic.cpp)

raja_add_executable(
  NAME resource-forall
  SOURCES resource-forall.cpp)

raja_add_executable(
  NAME tut_daxpy
  SOURCES tut_daxpy.cpp)

raja_add_executable(
  NAME tut_add-vectors
  SOURCES tut_add-vectors.cpp)

raja_add_executable(
  NAME tut_dot-product
  SOURCES tut_dot-product.cpp)

raja_add_executable(
  NAME tut_matrix-multiply
  SOURCES tut_matrix-multiply.cpp)

raja_add_executable(
  NAME tut_nested-loop-reorder
  SOURCES tut_nested-loop-reorder.cpp)

raja_add_executable(
  NAME tut_vertexsum-coloring
  SOURCES tut_vertexsum-coloring.cpp)

raja_add_executable(
  NAME teams_flatten
  SOURCES teams_flatten.cpp)

raja_add_executable(
  NAME teams_reductions
  SOURCES teams_reductions.cpp)

raja_add_executable(
  NAME resource-runtime-teams
  SOURCES resource-runtime-teams.cpp)

raja_add_executable(
  NAME tut_atomic-histogram
  SOURCES tut_atomic-histogram.cpp)

raja_add_executable(
  NAME tut_offset-layout
  SOURCES tut_offset-layout.cpp)

raja_add_executable(
  NAME tut_batched-matrix-multiply
  SOURCES tut_batched-matrix-multiply.cpp)

raja_add_executable(
  NAME tut_halo-exchange
  SOURCES tut_halo-exchange.cpp)

raja_add_executable(
  NAME pi-reduce_vs_atomic
  SOURCES pi-reduce_vs_atomic.cpp)

raja_add_executable(
  NAME raja-teams
  SOURCES raja-teams.cpp)

raja_add_executable(
  NAME teams_matrix-multiply
  SOURCES teams_matrix-multiply.cpp)

raja_add_executable(
  NAME jacobi
  SOURCES jacobi.cpp)

raja_add_executable(
  NAME red-black-gauss-seidel
  SOURCES red-black-gauss-seidel.cpp)

raja_add_executable(
  NAME wave-eqn
  SOURCES wave-eqn.cpp)

raja_add_executable(
  NAME multiview
  SOURCES multiview.cpp)

if(RAJA_ENABLE_TARGET_OPENMP)
  #  raja_add_executable(
  #    NAME target-kernel
  #    SOURCES omp-target-kernel.cpp)
  #
  #  raja_add_executable(
  #    NAME omp-target-ltimes
  #    SOURCES omp-target-ltimes.cpp)
endif()

raja_add_executable(
  NAME kernel-dynamic-tile
  SOURCES kernel-dynamic-tile.cpp)

raja_add_executable(
  NAME resource-kernel
  SOURCES resource-kernel.cpp)

raja_add_executable(
  NAME resource-teams
  SOURCES resource-teams.cpp)

add_subdirectory(plugin)
