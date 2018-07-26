###############################################################################
#
# Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-689114
#
# All rights reserved.
#
# This file is part of RAJA.
#
# For details about use and distribution, please read RAJA/LICENSE.
#
###############################################################################

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
  NAME tut_indexset-segments
  SOURCES tut_indexset-segments.cpp)

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
  NAME tut_reductions
  SOURCES tut_reductions.cpp)
  
raja_add_executable(
  NAME tut_scan
  SOURCES tut_scan.cpp)

raja_add_executable(
  NAME tut_atomic-binning
  SOURCES tut_atomic-binning.cpp)

raja_add_executable(
  NAME tut_offset-layout
  SOURCES tut_offset-layout.cpp)

raja_add_executable(
  NAME tut_batched-matrix-multiply
  SOURCES tut_batched-matrix-multiply.cpp)

raja_add_executable(
  NAME pi-reduce_vs_atomic
  SOURCES pi-reduce_vs_atomic.cpp)

raja_add_executable(
  NAME jacobi
  SOURCES jacobi.cpp)

raja_add_executable(
  NAME red-black-gauss-seidel
  SOURCES red-black-gauss-seidel.cpp)

raja_add_executable(
  NAME wave-eqn
  SOURCES wave-eqn.cpp)
  
  
if(ENABLE_CUDA)
raja_add_executable(
  NAME cuda-shmem-ltimes
  SOURCES cuda-shmem-ltimes.cpp)
endif()

raja_add_executable(
  NAME cpu-shmem-ltimes
  SOURCES cpu-shmem-ltimes.cpp)
