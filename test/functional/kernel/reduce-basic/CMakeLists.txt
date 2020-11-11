###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(LOCTYPES Min2D)

list(APPEND KERNEL_LOC_BACKENDS Sequential)

#if(RAJA_ENABLE_OPENMP)
#  list(APPEND KERNEL_LOC_BACKENDS OpenMP)
#endif()

#if(RAJA_ENABLE_TBB)
#  list(APPEND KERNEL_LOC_BACKENDS TBB)
#endif()


#
# Generate kernel region basic tests for each enabled RAJA back-end.
#
foreach( LOC_BACKEND ${KERNEL_LOC_BACKENDS} )
  foreach( LOC_TYPE ${LOCTYPES} )
    configure_file( test-kernel-loc.cpp.in
                    test-kernel-loc-${LOC_TYPE}-${LOC_BACKEND}.cpp )
    raja_add_test( NAME test-kernel-loc-${LOC_TYPE}-${LOC_BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel-loc-${LOC_TYPE}-${LOC_BACKEND}.cpp )

    target_include_directories(test-kernel-loc-${LOC_TYPE}-${LOC_BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( KERNEL_LOC_BACKENDS )
unset( LOCTYPES )


#
# Generate kernel loc tests for CUDA.
#
#if(RAJA_ENABLE_OPENMP)

#set(REGION_BACKEND OpenMP)

#configure_file( test-kernel-region-sync.cpp.in
#                test-kernel-region-sync-${REGION_BACKEND}.cpp )
#raja_add_test( NAME test-kernel-region-sync-${REGION_BACKEND}
#               SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel-region-sync-${REGION_BACKEND}.cpp )

#target_include_directories(test-kernel-region-sync-${REGION_BACKEND}.exe
#                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)

#unset(REGION_BACKEND)

#endif()
