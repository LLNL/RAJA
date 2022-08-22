###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
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
macro( buildfunctionalworkgrouptest TESTNAME SUBTESTNAMES DISPATCHERS BACKENDS )
  foreach( BACKEND ${BACKENDS} )
    foreach( DISPATCHER ${DISPATCHERS} )
      foreach( SUBTESTNAME ${SUBTESTNAMES} )

        if(${DISPATCHER} IN_LIST INDIRECT_DISPATCHERS)
          set(RAJA_USING_INDIRECT_DISPATCHER TRUE)
        endif()

        if(${BACKEND} STREQUAL "Hip" AND RAJA_USING_INDIRECT_DISPATCHER)
          set(RAJA_USING_HIP_INDIRECT_DISPATCHER TRUE)
        endif()

        # skip hip tests with indirect dispatchers
        if(RAJA_ENABLE_HIP_INDIRECT_FUNCTION_CALL OR NOT RAJA_USING_HIP_INDIRECT_DISPATCHER)
          configure_file( test-workgroup-${TESTNAME}.cpp.in
                          test-workgroup-${TESTNAME}-${SUBTESTNAME}-${DISPATCHER}-${BACKEND}.cpp )

          raja_add_test( NAME test-workgroup-${TESTNAME}-${SUBTESTNAME}-${DISPATCHER}-${BACKEND}
                         SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-workgroup-${TESTNAME}-${SUBTESTNAME}-${DISPATCHER}-${BACKEND}.cpp )

          target_include_directories( test-workgroup-${TESTNAME}-${SUBTESTNAME}-${DISPATCHER}-${BACKEND}.exe
                                      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests )
        endif()

        unset(RAJA_USING_HIP_INDIRECT_DISPATCHER)
        unset(RAJA_USING_INDIRECT_DISPATCHER)

      endforeach()
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


set(DISPATCHERS IndirectFunction IndirectVirtual Direct)
set(INDIRECT_DISPATCHERS IndirectFunction IndirectVirtual)


set(Ordered_SUBTESTS Single MultipleReuse)
buildfunctionalworkgrouptest(Ordered "${Ordered_SUBTESTS}" "${DISPATCHERS}" "${BACKENDS}")

set(Unordered_SUBTESTS Single MultipleReuse)
buildfunctionalworkgrouptest(Unordered "${Unordered_SUBTESTS}" "${DISPATCHERS}" "${BACKENDS}")

unset(BACKENDS)

#
# If building a subset of openmp target tests, add tests to build here.
#
if(RAJA_ENABLE_TARGET_OPENMP)
  if(RAJA_TEST_OPENMP_TARGET_SUBSET)

    set(BACKENDS OpenMPTarget)
    buildfunctionalworkgrouptest(Unordered "${Unordered_SUBTESTS}" "${DISPATCHERS}" "${BACKENDS}")

  endif()
endif()

unset(DISPATCHERS)
unset(INDIRECT_DISPATCHERS)
unset(BACKENDS)
unset(Ordered_SUBTESTS)
unset(Unordered_SUBTESTS)
