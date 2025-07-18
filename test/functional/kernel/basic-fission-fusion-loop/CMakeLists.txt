###############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# Generate tests for each enabled RAJA back-end.
#
# Note: KERNEL_BACKENDS is defined in ../CMakeLists.txt
#

##
## Enable OpenMP Target tests when support for fission-fusion is fixed
##
if(RAJA_ENABLE_TARGET_OPENMP)
  if(RAJA_TEST_OPENMP_TARGET_SUBSET)
    list(REMOVE_ITEM KERNEL_BACKENDS OpenMPTarget)
  endif()
endif()

foreach( BACKEND ${KERNEL_BACKENDS} )
      configure_file( test-kernel-basic-fission-fusion-loop.cpp.in
                      test-kernel-basic-fission-fusion-loop-${BACKEND}.cpp )
      raja_add_test( NAME test-kernel-basic-fission-fusion-loop-${BACKEND}
                     SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel-basic-fission-fusion-loop-${BACKEND}.cpp )
      target_include_directories(test-kernel-basic-fission-fusion-loop-${BACKEND}.exe
                                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()
