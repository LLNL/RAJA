###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of test types for future expansion, if needed. 
#
set(TESTTYPES ForICount TileTCount)
set(TILESIZES 8 32)

#
# Generate tests for each enabled RAJA back-end. 
# 
# Note: KERNEL_BACKENDS is defined in ../CMakeLists.txt
#
foreach( BACKEND ${KERNEL_BACKENDS} )
  foreach( TESTTYPE ${TESTTYPES} )
    foreach( TILESIZE ${TILESIZES} )
      configure_file( test-kernel-single-loop-tile-count.cpp.in
                      test-kernel-single-loop-${TESTTYPE}-${TILESIZE}-${BACKEND}.cpp )
      raja_add_test( NAME test-kernel-single-loop-${TESTTYPE}-${TILESIZE}-${BACKEND}
                     SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel-single-loop-${TESTTYPE}-${TILESIZE}-${BACKEND}.cpp )
      target_include_directories(test-kernel-single-loop-${TESTTYPE}-${TILESIZE}-${BACKEND}.exe
                                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    endforeach()
  endforeach()
endforeach()

unset( TILESIZES )
unset( TESTTYPES )
