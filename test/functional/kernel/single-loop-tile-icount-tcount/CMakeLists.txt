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

##
## Disable OpenMP Target tests, which cause front-end crash in LLVM based compilers
##
if(RAJA_ENABLE_TARGET_OPENMP)
  if(RAJA_TEST_OPENMP_TARGET_SUBSET)   
    list(REMOVE_ITEM KERNEL_BACKENDS OpenMPTarget)
  endif() 
endif()

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
