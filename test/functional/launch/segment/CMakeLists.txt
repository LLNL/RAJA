###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of segment types for generating test files.
#
#  TODO: Support list segments with RAJA Launch
#
set(SEGTYPES ListSegment RangeSegment RangeStrideSegment)


#
# Generate tests for each enabled RAJA back-end.
#
#

foreach( BACKEND ${LAUNCH_BACKENDS} )
  foreach( SEGTYPES ${SEGTYPES} )
    configure_file( test-launch-segment.cpp.in
                    test-launch-segment-${SEGTYPES}-${BACKEND}.cpp )
    raja_add_test( NAME test-launch-segment-${SEGTYPES}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-launch-segment-${SEGTYPES}-${BACKEND}.cpp )

    target_include_directories(test-launch-segment-${SEGTYPES}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( SEGTYPES )
