###############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# Generate tests for each enabled RAJA back-end.
#
#

foreach( BACKEND ${LAUNCH_UNCHECKED_BACKENDS} )
    configure_file( test-launch-nested-tile-unchecked.cpp.in
                    test-launch-nested-Tile-Unchecked-${BACKEND}.cpp )
    raja_add_test( NAME test-launch-nested-Tile-Unchecked-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-launch-nested-Tile-Unchecked-${BACKEND}.cpp )

    target_include_directories(test-launch-nested-Tile-Unchecked-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()

