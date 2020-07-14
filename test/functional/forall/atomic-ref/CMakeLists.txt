###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# Generate atomicref math tests for each enabled RAJA back-end.
#
# Note: FORALL_ATOMIC_BACKENDS is defined in ../CMakeLists.txt
#
foreach( ATOMIC_BACKEND ${FORALL_ATOMIC_BACKENDS} )
  configure_file( test-forall-atomicref-math.cpp.in
                  test-forall-atomicref-math-${ATOMIC_BACKEND}.cpp )
  raja_add_test( NAME test-forall-atomicref-math-${ATOMIC_BACKEND}
                 SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-forall-atomicref-math-${ATOMIC_BACKEND}.cpp )

  target_include_directories(test-forall-atomicref-math-${ATOMIC_BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()

#
# Generate atomicref other tests for each enabled RAJA back-end.
#
# Note: FORALL_ATOMIC_BACKENDS is defined in ../CMakeLists.txt
#
foreach( ATOMIC_BACKEND ${FORALL_ATOMIC_BACKENDS} )
  configure_file( test-forall-atomicref-other.cpp.in
                  test-forall-atomicref-other-${ATOMIC_BACKEND}.cpp )
  raja_add_test( NAME test-forall-atomicref-other-${ATOMIC_BACKEND}
                 SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-forall-atomicref-other-${ATOMIC_BACKEND}.cpp )

  target_include_directories(test-forall-atomicref-other-${ATOMIC_BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()
