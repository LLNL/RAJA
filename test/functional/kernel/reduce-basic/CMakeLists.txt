###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(LOCTYPES Min2D Max2D Min2DView Max2DView Min2DViewTuple Max2DViewTuple)

list(APPEND KERNEL_LOC_BACKENDS Sequential)

if(RAJA_ENABLE_OPENMP)
  list(APPEND KERNEL_LOC_BACKENDS OpenMP)
endif()

if(RAJA_ENABLE_TBB)
  list(APPEND KERNEL_LOC_BACKENDS TBB)
endif()


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
# Generate kernel loc tests for GPU.
#
if(RAJA_ENABLE_CUDA OR RAJA_ENABLE_HIP)

set(LOCTYPES Min2DViewTuple Max2DViewTuple)

if(RAJA_ENABLE_CUDA)
  list(APPEND KERNEL_LOC_BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND KERNEL_LOC_BACKENDS Hip)
endif()

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

endif() # end GPU backends
