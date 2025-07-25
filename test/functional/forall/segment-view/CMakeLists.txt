###############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of segment types for generating test files.
#
set(SEGVIEWTYPES ListSegmentView RangeSegmentView RangeSegment2DView RangeStrideSegmentView)

#
# Generate tests for each enabled RAJA back-end.
#
# Note: FORALL_BACKENDS is defined in ../CMakeLists.txt
#
foreach( BACKEND ${FORALL_BACKENDS} )
  foreach( SEGVIEWTYPE ${SEGVIEWTYPES} )
    configure_file( test-forall-segment-view.cpp.in
                    test-forall-${SEGVIEWTYPE}-${BACKEND}.cpp )
    raja_add_test( NAME test-forall-${SEGVIEWTYPE}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-forall-${SEGVIEWTYPE}-${BACKEND}.cpp )

    target_include_directories(test-forall-${SEGVIEWTYPE}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( SEGVIEWTYPES )
