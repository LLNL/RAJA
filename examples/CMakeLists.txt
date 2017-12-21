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
  NAME ex5-vertexsum-coloring
  SOURCES ex5-vertexsum-coloring.cpp)

raja_add_executable(
  NAME ex4-reductions
  SOURCES ex4-reductions.cpp)
  
raja_add_executable(
  NAME ex5-jacobi
  SOURCES ex5-jacobi.cpp)

raja_add_executable(
  NAME ex6-indexset-segments
  SOURCES ex6-indexset-segments.cpp)

raja_add_executable(
  NAME ex7-red-black-gauss-seidel
  SOURCES ex7-red-black-gauss-seidel.cpp)

raja_add_executable(
  NAME ex8-wave-eqn
  SOURCES ex8-wave-eqn.cpp)

raja_add_executable(
  NAME ex10-nested-loop-reorder
  SOURCES ex10-nested-loop-reorder.cpp)

raja_add_executable(
  NAME ex11-nested-loop-reorder-advanced
  SOURCES ex11-nested-loop-reorder-advanced.cpp)

raja_add_executable(
  NAME ex12-pi-atomic
  SOURCES ex12-pi-atomic.cpp)

raja_add_executable(
  NAME ex13-scan
  SOURCES ex13-scan.cpp)

