###############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
###############################################################################

raja_add_executable(
  NAME ex_1_add-vectors
  SOURCES ex_1_add-vectors.cpp)

raja_add_executable(
  NAME ex_2_dot-product
  SOURCES ex_2_dot-product.cpp)

raja_add_executable(
  NAME ex_3_indexset-segments
  SOURCES ex_3_indexset-segments.cpp)

raja_add_executable(
  NAME ex_4_atomic-binning
  SOURCES ex_4_atomic-binning.cpp)

raja_add_executable(
  NAME ex_5_batched-matrix-multiply
  SOURCES ex_5_batched-matrix-multiply.cpp)

raja_add_executable(
  NAME ex_6_matrix-multiply
  SOURCES ex_6_matrix-multiply.cpp)

raja_add_executable(
  NAME ex_7_tiled-matrix-transpose
  SOURCES ex_7_tiled-matrix-transpose.cpp)

raja_add_executable(
  NAME ex_8_matrix-transpose-local-array
  SOURCES ex_8_matrix-transpose-local-array.cpp)
