###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of reduction types for generating test files.
#
set(REDUCETYPES ReduceSum ReduceMin ReduceMax ReduceMinLoc ReduceMaxLoc)

if(RAJA_ENABLE_TARGET_OPENMP)
  if(RAJA_TEST_OPENMP_TARGET_SUBSET)
    list(REMOVE_ITEM FORALL_BACKENDS OpenMPTarget)
  endif()
endif()

#
# While we're adding SYCL tests, enable it for each test set like this.
#
# Eventually, remove this and enable in the top-level CMakeLists.txt file.
#
#if(RAJA_ENABLE_SYCL)
#  list(APPEND FORALL_BACKENDS Sycl)
#endif()

#
# Generate tests for each enabled RAJA back-end
#
# Note: FORALL_BACKENDS is defined in ../CMakeLists.txt
#
foreach( BACKEND ${FORALL_BACKENDS} )
  foreach( REDUCETYPE ${REDUCETYPES} )
    configure_file( test-forall-indexset-multiple-reduce.cpp.in
                    test-forall-indexset-multiple-${REDUCETYPE}-${BACKEND}.cpp )
    raja_add_test( NAME test-forall-indexset-multiple-${REDUCETYPE}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-forall-indexset-multiple-${REDUCETYPE}-${BACKEND}.cpp )

    target_include_directories(test-forall-indexset-multiple-${REDUCETYPE}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( REDUCETYPES )
