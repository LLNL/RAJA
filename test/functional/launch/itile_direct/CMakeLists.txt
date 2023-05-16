###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# Generate tests for each enabled RAJA back-end.
#
#

foreach( BACKEND ${LAUNCH_BACKENDS} )
    configure_file( test-launch-nested-itile-direct.cpp.in
                    test-launch-nested-iTile-Direct-${BACKEND}.cpp )
    raja_add_test( NAME test-launch-nested-iTile-Direct-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-launch-nested-iTile-Direct-${BACKEND}.cpp )

    target_include_directories(test-launch-nested-iTile-Direct-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()

