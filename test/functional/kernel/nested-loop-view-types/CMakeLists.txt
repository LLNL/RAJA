###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of test types for future expansion, if needed. 
#
#set(TESTTYPES PermutedView OffsetView PermutedOffsetView)
set(TESTTYPES PermutedView2D PermutedView3D)

#
# Generate tests for each enabled RAJA back-end. 
# 
# Note: KERNEL_BACKENDS is defined in ../CMakeLists.txt
#
foreach( BACKEND ${KERNEL_BACKENDS} )
  foreach( TESTTYPE ${TESTTYPES} )
    configure_file( test-kernel-nested-loop-view.cpp.in
                    test-kernel-nested-loop-${TESTTYPE}-${BACKEND}.cpp )
    raja_add_test( NAME test-kernel-nested-loop-${TESTTYPE}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel-nested-loop-${TESTTYPE}-${BACKEND}.cpp )
    target_include_directories(test-kernel-nested-loop-${TESTTYPE}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( TEST_TYPES )
