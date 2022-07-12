###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# Generate tests for each enabled RAJA back-end. 
# 
# Note: KERNEL_BACKENDS is defined in ../CMakeLists.txt
#

if(RAJA_ENABLE_TARGET_OPENMP)
  if(RAJA_TEST_OPENMP_TARGET_SUBSET)
    list(REMOVE_ITEM KERNEL_BACKENDS OpenMPTarget)
  endif()
endif()

#
# Remove SYCL until kernel reduction support is added
#
if(RAJA_ENABLE_SYCL)
  list(REMOVE_ITEM KERNEL_BACKENDS Sycl)
endif()

#
# While we're adding SYCL tests, enable it for each test set like this.
#
# Eventually, remove this and enable in the top-level CMakeLists.txt file.
#
#if(RAJA_ENABLE_SYCL)
#  list(APPEND KERNEL_BACKENDS Sycl)
#endif()

foreach( BACKEND ${KERNEL_BACKENDS} )
  configure_file( test-kernel-nested-loop-segments.cpp.in
                  test-kernel-nested-loop-segments-${BACKEND}.cpp )
  raja_add_test( NAME test-kernel-nested-loop-segments-${BACKEND}
                 SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel-nested-loop-segments-${BACKEND}.cpp )

  target_include_directories(test-kernel-nested-loop-segments-${BACKEND}.exe
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()
