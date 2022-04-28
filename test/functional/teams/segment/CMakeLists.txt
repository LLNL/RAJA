###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of segment types for generating test files.
#
#set(SEGTYPES ListSegment RangeSegment)
set(SEGTYPES RangeSegment RangeStrideSegment)


#
# Generate tests for each enabled RAJA back-end.
#
#

foreach( BACKEND ${TEAMS_BACKENDS} )
  foreach( SEGTYPES ${SEGTYPES} )
    configure_file( test-teams-segment.cpp.in
                    test-teams-segment-${SEGTYPES}-${BACKEND}.cpp )
    raja_add_test( NAME test-teams-segment-${SEGTYPES}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-teams-segment-${SEGTYPES}-${BACKEND}.cpp )

    target_include_directories(test-teams-segment-${SEGTYPES}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( SEGTYPES )
