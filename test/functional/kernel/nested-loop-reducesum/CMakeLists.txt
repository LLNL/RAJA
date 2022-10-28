###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(NESTED_LOOPTYPES ReduceSum BlockReduceSum)

set( USE_RESOURCE "-resource-" "-" )

#
# If building a subset of openmp target tests, remove the back-end from
# from the list of tests to generate here.
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
# Generate kernel region basic tests for each enabled RAJA back-end.
#
foreach( NESTED_LOOP_BACKEND ${KERNEL_BACKENDS} )
  foreach( RESOURCE ${USE_RESOURCE} )
    foreach( NESTED_LOOP_TYPE ${NESTED_LOOPTYPES} )
      if( ${NESTED_LOOP_TYPE} STREQUAL "ReduceSum" OR # allow all ReduceSum tests
          ((${NESTED_LOOP_BACKEND} STREQUAL "Sequential" OR ${NESTED_LOOP_BACKEND} STREQUAL "Cuda" OR ${NESTED_LOOP_BACKEND} STREQUAL "Hip" ) AND ${NESTED_LOOP_TYPE} STREQUAL "BlockReduceSum") # allow only certain BlockReduceSum tests
        )
        # Note on BlockReduceSum: Inherent kernel reduction functionality does not exist for - OpenMP, OpenMPTarget, and TBB.
        configure_file( test-kernel-nested-loop.cpp.in
                        test-kernel${RESOURCE}nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}.cpp )

        raja_add_test( NAME test-kernel${RESOURCE}nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}
            SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel${RESOURCE}nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}.cpp )

        target_include_directories(test-kernel${RESOURCE}nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}.exe
                                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
      endif()

    endforeach()
  endforeach()
endforeach()

unset( NESTED_LOOPTYPES )

#
# If building a subset of openmp target tests, add tests to build here.
#
if(RAJA_ENABLE_TARGET_OPENMP)
  if(RAJA_TEST_OPENMP_TARGET_SUBSET)

    set(NESTED_LOOP_BACKEND OpenMPTarget)
    set(NESTED_LOOPTYPES ReduceSum)

    set(RESOURCE "-")
    foreach( NESTED_LOOP_TYPE ${NESTED_LOOPTYPES} )
      configure_file( test-kernel-nested-loop.cpp.in
                      test-kernel-nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}.cpp )
      raja_add_test( NAME test-kernel-nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}
                     SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel-nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}.cpp )

      target_include_directories(test-kernel-nested-loop-${NESTED_LOOP_TYPE}-${NESTED_LOOP_BACKEND}.exe
                                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()

  endif()
endif()

unset( NESTED_LOOPTYPES )
