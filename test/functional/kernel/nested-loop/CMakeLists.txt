###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(NESTED_LOOPTYPES Double )

list(APPEND KERNEL_NESTED_LOOP_BACKENDS Sequential)

if(RAJA_ENABLE_OPENMP)
  list(APPEND KERNEL_NESTED_LOOP_BACKENDS OpenMP)
endif()
#
#if(RAJA_ENABLE_TBB)
#  list(APPEND KERNEL_NESTED_LOOP_BACKENDS TBB)
#endif()
#
#if(RAJA_ENABLE_CUDA)
#  list(APPEND KERNEL_NESTED_LOOP_BACKENDS Cuda)
#endif()
#
#if(RAJA_ENABLE_HIP)
#  list(APPEND KERNEL_NESTED_LOOP_BACKENDS Hip)
#endif()
#
#if(RAJA_ENABLE_TARGET_OPENMP)
#  list(APPEND KERNEL_NESTED_LOOP_BACKENDS OpenMPTarget)
#endif()


#
# Generate kernel region basic tests for each enabled RAJA back-end.
#
foreach( NESTED_LOOP_BACKEND ${KERNEL_NESTED_LOOP_BACKENDS} )
  foreach( NESTED_LOOP_TYPE ${NESTED_LOOPTYPES} )
    configure_file( test-kernel-nested-loop.cpp.in
                    test-kernel-nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}.cpp )
    raja_add_test( NAME test-kernel-nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel-nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}.cpp )

    target_include_directories(test-kernel-nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( KERNEL_NESTED_LOOP_BACKENDS )
unset( NESTED_LOOPTYPES )
