###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(NESTED_LOOPTYPES Basic MultiLambda MultiLambdaParam)

#
# Generate kernel region basic tests for each enabled RAJA back-end.
#
foreach( NESTED_LOOP_BACKEND ${KERNEL_BACKENDS} )
  foreach( NESTED_LOOP_TYPE ${NESTED_LOOPTYPES} )
    configure_file( test-kernel-nested-loop.cpp.in
                    test-kernel-nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}.cpp )

    #Some tests are known to fail for Hip, mark those tests (Will not be run in Gitlab CI)
    if(${NESTED_LOOP_BACKEND} STREQUAL "Hip" AND ${NESTED_LOOP_TYPE} STREQUAL "MultiLambda")
        raja_add_test( NAME test-kernel-nested-loop-Known-Hip-Failure-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel-nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}.cpp )

        target_include_directories(test-kernel-nested-loop-Known-Hip-Failure-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    else()
        raja_add_test( NAME test-kernel-nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel-nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}.cpp )

        target_include_directories(test-kernel-nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    endif()

  endforeach()
endforeach()

unset( NESTED_LOOPTYPES )
