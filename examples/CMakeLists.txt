###############################################################################
#
# Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
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
  NAME ex0-daxpy
  SOURCES ex0-daxpy.cpp)

raja_add_executable(
  NAME ex1-add-vectors
  SOURCES ex1-add-vectors.cpp)

raja_add_executable(
  NAME ex2-dot-product
  SOURCES ex2-dot-product.cpp)

raja_add_executable(
  NAME ex3-indexset-segments
  SOURCES ex3-indexset-segments.cpp)

raja_add_executable(
  NAME ex4-matrix-multiply
  SOURCES ex4-matrix-multiply.cpp)

raja_add_executable(
  NAME ex5-nested-loop-reorder
  SOURCES ex5-nested-loop-reorder.cpp)

raja_add_executable(
  NAME ex6-vertexsum-coloring
  SOURCES ex6-vertexsum-coloring.cpp)

raja_add_executable(
  NAME ex7-reductions
  SOURCES ex7-reductions.cpp)
  
raja_add_executable(
  NAME ex8-pi-reduce_vs_atomic
  SOURCES ex8-pi-reduce_vs_atomic.cpp)

raja_add_executable(
  NAME ex9-scan
  SOURCES ex9-scan.cpp)

raja_add_executable(
  NAME ex10-binning.cpp
  SOURCES ex10-binning.cpp)

raja_add_executable(
  NAME ex11-jacobi
  SOURCES ex11-jacobi.cpp)

raja_add_executable(
  NAME ex12-red-black-gauss-seidel
  SOURCES ex12-red-black-gauss-seidel.cpp)

raja_add_executable(
  NAME ex13-wave-eqn
  SOURCES ex13-wave-eqn.cpp)
