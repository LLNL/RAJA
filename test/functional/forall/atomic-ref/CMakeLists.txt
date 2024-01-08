###############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of atomic test types for generating test files.
#
set(TESTTYPES AtomicRefAdd AtomicRefSub AtomicRefLoadStore AtomicRefCAS AtomicRefMinMax AtomicRefLogical)

#
# If building a subset of openmp target tests, remove the back-end from
# from the list of tests to generate here.
#
if(RAJA_ENABLE_TARGET_OPENMP)
  if(RAJA_TEST_OPENMP_TARGET_SUBSET)
    list(REMOVE_ITEM FORALL_ATOMIC_BACKENDS OpenMPTarget)
  endif()
endif()

#
# Generate atomicref tests for each enabled RAJA back-end.
#
# Note: FORALL_ATOMIC_BACKENDS is defined in ../CMakeLists.txt
#
foreach( ATOMIC_BACKEND ${FORALL_ATOMIC_BACKENDS} )
  foreach( TEST ${TESTTYPES} )
    configure_file( test-forall-atomicref.cpp.in
                    test-forall-${TEST}-${ATOMIC_BACKEND}.cpp )
    raja_add_test( NAME test-forall-${TEST}-${ATOMIC_BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-forall-${TEST}-${ATOMIC_BACKEND}.cpp )

    target_include_directories(test-forall-${TEST}-${ATOMIC_BACKEND}.exe
                                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( TESTTYPES )

#
# If building a subset of openmp target tests, add tests to build here.
#
if(RAJA_ENABLE_TARGET_OPENMP)
  if(RAJA_TEST_OPENMP_TARGET_SUBSET)

    set(ATOMIC_BACKEND OpenMPTarget)
    set(TESTTYPES AtomicRefAdd AtomicRefCAS)
    
    foreach( TEST ${TESTTYPES} )
      configure_file( test-forall-atomicref.cpp.in
                      test-forall-${TEST}-${ATOMIC_BACKEND}.cpp )
      raja_add_test( NAME test-forall-${TEST}-${ATOMIC_BACKEND}
                     SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-forall-${TEST}-${ATOMIC_BACKEND}.cpp )

      target_include_directories(test-forall-${TEST}-${ATOMIC_BACKEND}.exe
                                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    endforeach()

    unset( TESTTYPES )

  endif()
endif()


