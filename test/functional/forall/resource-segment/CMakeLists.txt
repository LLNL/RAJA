###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
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
# Note: FORALL_BACKENDS is defined in ../CMakeLists.txt
#
foreach( BACKEND ${FORALL_BACKENDS} )
  foreach( SEGTYPE ${SEGTYPES} )
    configure_file( test-forall-resource-segment.cpp.in
                    test-forall-resource-${SEGTYPE}-${BACKEND}.cpp )
    raja_add_test( NAME test-forall-resource-${SEGTYPE}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-forall-resource-${SEGTYPE}-${BACKEND}.cpp )

    target_include_directories(test-forall-resource-${SEGTYPE}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( SEGTYPES )
