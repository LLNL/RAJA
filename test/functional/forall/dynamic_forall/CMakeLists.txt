###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of segment types for generating test files.
#
set(TEST_TYPES DynamicRangeSegmentForall)

foreach( BACKEND ${DYNAMIC_BACKENDS} )
  foreach( TESTTYPE ${TEST_TYPES} )
    configure_file( test-dynamic-forall.cpp.in
                    test-dynamic-forall-${TESTTYPE}-${BACKEND}.cpp )
    raja_add_test( NAME test-dynamic-forall-${TESTTYPE}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-dynamic-forall-${TESTTYPE}-${BACKEND}.cpp )

    target_include_directories(test-dynamic-forall-${TESTTYPE}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( TEST_TYPES )
