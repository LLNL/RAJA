###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# Generate tests for each enabled RAJA back-end. 
# 
# Note: KERNEL_BACKENDS is defined in ../CMakeLists.txt
#
foreach( BACKEND ${KERNEL_BACKENDS} )
      configure_file( test-basic-fision-fussion-loop.cpp.in
                      test-basic-fision-fussion-loop-${BACKEND}.cpp )
      raja_add_test( NAME test-basic-fision-fussion-loop-${BACKEND}
                     SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-basic-fision-fussion-loop-${BACKEND}.cpp )
      target_include_directories(test-basic-fision-fussion-loop-${BACKEND}.exe
                                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()

unset( TEST_TYPES )