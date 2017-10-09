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
  NAME example-pi
  SOURCES example-pi.cpp)

raja_add_executable(
  NAME example-add-vectors
  SOURCES example-add-vectors.cpp)

raja_add_executable(
  NAME example-matrix-multiply
  SOURCES example-matrix-multiply.cpp)

raja_add_executable(
  NAME example-jacobi
  SOURCES example-jacobi.cpp)

raja_add_executable(
  NAME example-wave
  SOURCES example-wave.cpp)

raja_add_executable(
  NAME example-custom-index
  SOURCES example-custom-index.cpp)

raja_add_executable(
  NAME example-gauss-seidel
  SOURCES example-gauss-seidel.cpp)
