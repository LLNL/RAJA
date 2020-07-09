###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
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


set(Constructor_SUBTESTS Single)
buildunitworkgrouptest(Constructor "${Constructor_SUBTESTS}" "${BACKENDS}")

set(Vtable_SUBTESTS Single)
buildunitworkgrouptest(Vtable      "${Vtable_SUBTESTS}"      "${Vtable_BACKENDS}")

set(Enqueue_SUBTESTS Single Multiple)
buildunitworkgrouptest(Enqueue     "${Enqueue_SUBTESTS}"     "${BACKENDS}")

set(WorkStorage_SUBTESTS Constructor Insert Iterator InsertCall Multiple)
buildunitworkgrouptest(WorkStorage "${WorkStorage_SUBTESTS}" "${WorkStorage_BACKENDS}")

unset(BACKENDS)
unset(Vtable_BACKENDS)
unset(WorkStorage_BACKENDS)

unset(Constructor_SUBTESTS)
unset(Vtable_SUBTESTS)
unset(Enqueue_SUBTESTS)
unset(WorkStorage_SUBTESTS)
