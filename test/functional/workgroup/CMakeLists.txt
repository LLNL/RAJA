###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# macro that generates test file and build target for each
# sub-test and backend
# this must be a macro or the linker variable set by FindHIP won't be set in
# the right scope and linking will fail with a weird error from
# hipcc_cmake_linker_helper because it expects the path to hipcc as the first
# argument
#
macro( buildunitworkgrouptest TESTNAME SUBTESTNAMES BACKENDS )
  foreach( BACKEND ${BACKENDS} )
    foreach( SUBTESTNAME ${SUBTESTNAMES} )
      configure_file( test-workgroup-${TESTNAME}.cpp.in
                      test-workgroup-${TESTNAME}-${SUBTESTNAME}-${BACKEND}.cpp )
      raja_add_test( NAME test-workgroup-${TESTNAME}-${SUBTESTNAME}-${BACKEND}
                     SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-workgroup-${TESTNAME}-${SUBTESTNAME}-${BACKEND}.cpp )

      target_include_directories(test-workgroup-${TESTNAME}-${SUBTESTNAME}-${BACKEND}.exe
                                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    endforeach()
  endforeach()
endmacro()


set(BACKENDS Sequential)

if(RAJA_ENABLE_TBB)
  list(APPEND BACKENDS TBB)
endif()

if(RAJA_ENABLE_OPENMP)
  list(APPEND BACKENDS OpenMP)
endif()

#
# If building a subset of openmp target tests, do not add the back-end to
# the list of tests to generate here.
#
if(RAJA_ENABLE_TARGET_OPENMP)
  if(NOT RAJA_TEST_OPENMP_TARGET_SUBSET)
    list(APPEND BACKENDS OpenMPTarget)
  endif() 
endif()

if(RAJA_ENABLE_CUDA)
  list(APPEND BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND BACKENDS Hip)
endif()


set(Ordered_SUBTESTS Single MultipleReuse)
buildunitworkgrouptest(Ordered "${Ordered_SUBTESTS}" "${BACKENDS}")

set(Unordered_SUBTESTS Single MultipleReuse)
buildunitworkgrouptest(Unordered "${Unordered_SUBTESTS}" "${BACKENDS}")

unset(BACKENDS)

#
# If building a subset of openmp target tests, add tests to build here.
#
if(RAJA_ENABLE_TARGET_OPENMP)
  if(RAJA_TEST_OPENMP_TARGET_SUBSET)

    set(BACKENDS OpenMPTarget)
    buildunitworkgrouptest(Unordered "${Unordered_SUBTESTS}" "${BACKENDS}")

  endif()
endif()

unset(BACKENDS)
unset(Ordered_SUBTESTS)
unset(Unordered_SUBTESTS)
