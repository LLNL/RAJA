###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

raja_add_test(
  NAME test-CombiningAdapter-1D
  SOURCES test-CombiningAdapter-1D.cpp)

raja_add_test(
  NAME test-CombiningAdapter-2D
  SOURCES test-CombiningAdapter-2D.cpp)

raja_add_test(
  NAME test-CombiningAdapter-3D
  SOURCES test-CombiningAdapter-3D.cpp)

raja_add_test(
  NAME test-PermutedCombiningAdapter-1D
  SOURCES test-PermutedCombiningAdapter-1D.cpp)

raja_add_test(
  NAME test-PermutedCombiningAdapter-2D
  SOURCES test-PermutedCombiningAdapter-2D.cpp)

raja_add_test(
  NAME test-PermutedCombiningAdapter-3D
  SOURCES test-PermutedCombiningAdapter-3D.cpp)
