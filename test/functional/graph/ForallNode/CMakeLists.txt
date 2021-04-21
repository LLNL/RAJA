###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of segment types for generating test files.
#
set(SEGTYPES ListSegment RangeSegment RangeStrideSegment)


#
# Generate tests for each enabled RAJA back-end.
#
# Note: GRAPH_FORALL_BACKENDS is defined in ../CMakeLists.txt
#
foreach(GRAPHBACKEND ${GRAPH_BACKENDS})
  foreach( BACKEND ${GRAPH_FORALL_BACKENDS} )
    foreach( SEGTYPE ${SEGTYPES} )
      configure_file( test-graph-ForallNode-segment.cpp.in
                      test-graph-${GRAPHBACKEND}-ForallNode-${SEGTYPE}-${BACKEND}.cpp )
      raja_add_test( NAME test-graph-${GRAPHBACKEND}-ForallNode-${SEGTYPE}-${BACKEND}
                     SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-graph-${GRAPHBACKEND}-ForallNode-${SEGTYPE}-${BACKEND}.cpp )

      target_include_directories(test-graph-${GRAPHBACKEND}-ForallNode-${SEGTYPE}-${BACKEND}.exe
                                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    endforeach()
  endforeach()
endforeach()

unset( SEGTYPES )
