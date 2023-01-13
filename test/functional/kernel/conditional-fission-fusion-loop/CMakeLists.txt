###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# Generate tests for each enabled RAJA back-end.
#
# Note: KERNEL_BACKENDS is defined in ../CMakeLists.txt
#

#
# While we're adding SYCL tests, enable it for each test set like this.
#
# Eventually, remove this and enable in the top-level CMakeLists.txt file.
#
#if(RAJA_ENABLE_SYCL)
#  list(APPEND KERNEL_BACKENDS Sycl)
#endif()

foreach( BACKEND ${KERNEL_BACKENDS} )
      configure_file( test-kernel-conditional-fission-fusion-loop.cpp.in
                      test-kernel-conditional-fission-fusion-loop-${BACKEND}.cpp )
      raja_add_test( NAME test-kernel-conditional-fission-fusion-loop-${BACKEND}
                     SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel-conditional-fission-fusion-loop-${BACKEND}.cpp )
      target_include_directories(test-kernel-conditional-fission-fusion-loop-${BACKEND}.exe
                                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()
