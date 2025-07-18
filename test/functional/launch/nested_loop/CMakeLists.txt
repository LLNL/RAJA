###############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of segment types for generating test files.
#
set(NESTEDTYPES Loop)

#
# Generate tests for each enabled RAJA back-end.
#
#

foreach( BACKEND ${LAUNCH_BACKENDS} )
  foreach( NESTEDTYPES ${NESTEDTYPES} )
    configure_file( test-launch-nested.cpp.in
                    test-launch-nested-${NESTEDTYPES}-${BACKEND}.cpp )
    raja_add_test( NAME test-launch-nested-${NESTEDTYPES}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-launch-nested-${NESTEDTYPES}-${BACKEND}.cpp )

    target_include_directories(test-launch-nested-${NESTEDTYPES}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( NESTEDTYPES )
