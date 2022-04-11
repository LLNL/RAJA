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
set(Vtable_BACKENDS Sequential)
set(WorkStorage_BACKENDS Sequential)

if(RAJA_ENABLE_TBB)
  list(APPEND BACKENDS TBB)
  list(APPEND Vtable_BACKENDS TBB)
endif()

if(RAJA_ENABLE_OPENMP)
  list(APPEND BACKENDS OpenMP)
  list(APPEND Vtable_BACKENDS OpenMP)
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  list(APPEND BACKENDS OpenMPTarget)
  list(APPEND Vtable_BACKENDS OpenMPTarget)
endif()

if(RAJA_ENABLE_CUDA)
  list(APPEND BACKENDS Cuda)
  list(APPEND Vtable_BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND BACKENDS Hip)
  if(RAJA_ENABLE_HIP_INDIRECT_FUNCTION_CALL)
    list(APPEND Vtable_BACKENDS Hip)
  endif()
endif()

# reduce travis build times with intel compiler
if(RAJA_TEST_EXHAUSTIVE OR NOT RAJA_COMPILER MATCHES "RAJA_COMPILER_Intel")
  set(Constructor_SUBTESTS Single)
  buildunitworkgrouptest(Constructor "${Constructor_SUBTESTS}" "${BACKENDS}")

  set(Enqueue_SUBTESTS Single Multiple)
  buildunitworkgrouptest(Enqueue     "${Enqueue_SUBTESTS}"     "${BACKENDS}")

  unset(Constructor_SUBTESTS)
  unset(Enqueue_SUBTESTS)
endif()

set(Vtable_SUBTESTS Single)
buildunitworkgrouptest(Vtable      "${Vtable_SUBTESTS}"      "${Vtable_BACKENDS}")

set(WorkStorage_SUBTESTS Constructor Iterator InsertCall Multiple)
buildunitworkgrouptest(WorkStorage "${WorkStorage_SUBTESTS}" "${WorkStorage_BACKENDS}")

unset(Vtable_SUBTESTS)
unset(WorkStorage_SUBTESTS)

unset(BACKENDS)
unset(Vtable_BACKENDS)
unset(WorkStorage_BACKENDS)
