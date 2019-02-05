###############################################################################
#
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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
  NAME tut_1_add-vectors
  SOURCES tut_1_add-vectors.cpp)

raja_add_executable(
  NAME tut_2_dot-product
  SOURCES tut_2_dot-product.cpp)

raja_add_executable(
  NAME tut_3_indexset-segments
  SOURCES tut_3_indexset-segments.cpp)

raja_add_executable(
  NAME tut_4_atomic-binning
  SOURCES tut_4_atomic-binning.cpp)

raja_add_executable(
  NAME tut_5_batched-matrix-multiply
  SOURCES tut_5_batched-matrix-multiply.cpp)

raja_add_executable(
  NAME tut_6_matrix-multiply
  SOURCES tut_6_matrix-multiply.cpp)

raja_add_executable(
  NAME tut_7_tiled-matrix-transpose
  SOURCES tut_7_tiled-matrix-transpose.cpp)

raja_add_executable(
  NAME tut_8_matrix-transpose-local-array
  SOURCES tut_8_matrix-transpose-local-array.cpp)
