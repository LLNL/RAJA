###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of segment types for generating test files.
#
set(SEGTYPE RangeSegment)

foreach( BACKEND ${DYNAMIC_BACKENDS} )
  foreach( TESTTYPE ${SEGTYPE} )
    configure_file( test-dynamic-forall-segments.cpp.in
                    test-dynamic-forall-segments-${SEGTYPE}-${BACKEND}.cpp )
    raja_add_test( NAME test-dynamic-forall-segments-${SEGTYPE}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-dynamic-forall-segments-${SEGTYPE}-${BACKEND}.cpp )

    target_include_directories(test-dynamic-forall-segments-${SEGTYPE}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( SEGTYPE )
