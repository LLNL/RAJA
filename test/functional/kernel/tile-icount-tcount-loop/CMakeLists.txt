###############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
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
  # using omp target crashes the compiler with this one 
  if( NOT ((BACKEND STREQUAL "OpenMPTarget")) )
    foreach( TESTTYPE ${TESTTYPES} )
      foreach( TILESIZE ${TILESIZES} )
        configure_file( test-kernel-tile-count-loop.cpp.in
                        test-kernel-${TESTTYPE}-${TILESIZE}-${BACKEND}-loop.cpp )
        raja_add_test( NAME test-kernel-${TESTTYPE}-${TILESIZE}-${BACKEND}-loop
                      SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel-${TESTTYPE}-${TILESIZE}-${BACKEND}-loop.cpp )
        target_include_directories(test-kernel-${TESTTYPE}-${TILESIZE}-${BACKEND}-loop.exe
                                  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
      endforeach()
    endforeach()
  endif()
endforeach()

unset( TILESIZES )
unset( TESTTYPES )
