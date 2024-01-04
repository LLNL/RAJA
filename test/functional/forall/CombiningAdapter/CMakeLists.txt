###############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of dimensions for generating test files.
#
set(DIMENSIONS 1D 2D 3D)

#
# Generate tests for each enabled RAJA back-end.
#
# Note: FORALL_BACKENDS is defined in ../CMakeLists.txt
#
foreach( BACKEND ${FORALL_BACKENDS} )
  foreach( DIMENSION ${DIMENSIONS} )
    configure_file( test-forall-CombiningAdapter.cpp.in
                    test-forall-CombiningAdapter-${DIMENSION}-${BACKEND}.cpp )
    raja_add_test( NAME test-forall-CombiningAdapter-${DIMENSION}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-forall-CombiningAdapter-${DIMENSION}-${BACKEND}.cpp )

    target_include_directories(test-forall-CombiningAdapter-${DIMENSION}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()
