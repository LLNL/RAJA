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

foreach( BACKEND ${LAUNCH_DIRECT_UNCHECKED_BACKENDS} )
    configure_file( test-launch-nested-tile-icount-tcount-direct-unchecked.cpp.in
                    test-launch-nested-Tile-iCount-tCount-DirectUnchecked-${BACKEND}.cpp )
    raja_add_test( NAME test-launch-nested-Tile-iCount-tCount-DirectUnchecked-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-launch-nested-Tile-iCount-tCount-DirectUnchecked-${BACKEND}.cpp )

    target_include_directories(test-launch-nested-Tile-iCount-tCount-DirectUnchecked-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()

