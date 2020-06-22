###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# Function that generates test file and build target for each
# sub-test and backend
#
function( buildunitworkgrouptest TESTNAME SUBTESTNAMES BACKENDS )
  foreach( BACKEND ${BACKENDS} )
    foreach( SUBTESTNAME ${SUBTESTNAMES} )
      configure_file( test-workgroup-${TESTNAME}.cpp.in
                      test-workgroup-${TESTNAME}-${SUBTESTNAME}-${BACKEND}.cpp )
      raja_add_test( NAME test-workgroup-${TESTNAME}-${SUBTESTNAME}-${BACKEND}
                     SOURCES test-workgroup-${TESTNAME}-${SUBTESTNAME}-${BACKEND}.cpp )

      target_include_directories(test-workgroup-${TESTNAME}-${SUBTESTNAME}-${BACKEND}.exe
                                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    endforeach()
  endforeach()
endfunction()


set(BACKENDS Sequential)

if(RAJA_ENABLE_TBB)
  list(APPEND BACKENDS TBB)
endif()

if(RAJA_ENABLE_OPENMP)
  list(APPEND BACKENDS OpenMP)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  list(APPEND BACKENDS OpenMPTarget)
endif()

if(RAJA_ENABLE_CUDA)
  list(APPEND BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND BACKENDS Hip)
endif()


set(Ordered_SUBTESTS Single Multiple)
buildunitworkgrouptest(Ordered "${Ordered_SUBTESTS}" "${BACKENDS}")

set(Unordered_SUBTESTS Single Multiple)
buildunitworkgrouptest(Unordered "${Unordered_SUBTESTS}" "${BACKENDS}")
