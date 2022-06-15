###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of segment types for generating test files.
#
set(TEST_TYPES BasicShared)

foreach( BACKEND ${TEAMS_BACKENDS} )
  foreach( TESTTYPE ${TEST_TYPES} )
    configure_file( test-teams.cpp.in
                    test-teams-${TESTTYPE}-${BACKEND}.cpp )
    raja_add_test( NAME test-teams-${TESTTYPE}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-teams-${TESTTYPE}-${BACKEND}.cpp )

    target_include_directories(test-teams-${TESTTYPE}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( TEST_TYPES )
