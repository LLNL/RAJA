###############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# Generate tests for each enabled RAJA back-end.
#
# Note: FORALL_ATOMIC_BACKENDS is defined in ../CMakeLists.txt
#
foreach( ATOMIC_BACKEND ${FORALL_ATOMIC_BACKENDS} )
  # Signed Tests
  configure_file( test-forall-atomic-basic.cpp.in
                  test-forall-atomic-basic-${ATOMIC_BACKEND}.cpp )
  raja_add_test( NAME test-forall-atomic-basic-${ATOMIC_BACKEND}
                 SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-forall-atomic-basic-${ATOMIC_BACKEND}.cpp )

  target_include_directories(test-forall-atomic-basic-${ATOMIC_BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  # Unsigned Tests
  configure_file( test-forall-atomic-basic-unsigned.cpp.in
                  test-forall-atomic-basic-unsigned-${ATOMIC_BACKEND}.cpp )
  raja_add_test( NAME test-forall-atomic-basic-unsigned-${ATOMIC_BACKEND}
                SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-forall-atomic-basic-unsigned-${ATOMIC_BACKEND}.cpp )

  target_include_directories(test-forall-atomic-basic-unsigned-${ATOMIC_BACKEND}.exe
                              PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()


set(ENABLE_LITMUS_TESTS OFF)

if(RAJA_ENABLE_DESUL_ATOMICS AND RAJA_ENABLE_CUDA)
    set(LITMUS_BACKEND "Cuda")
    set(ENABLE_LITMUS_TESTS ON)
endif()

if(RAJA_ENABLE_DESUL_ATOMICS AND RAJA_ENABLE_HIP)
    set(LITMUS_BACKEND "Hip")
    set(ENABLE_LITMUS_TESTS ON)
endif()

set(FORALL_LITMUS_TESTS
    mp # Message Passing
    sb # Store Buffer
    lb # Load Buffer
    store # Store
    read # Read
    write2x2 # 2+2 write
)

if (ENABLE_LITMUS_TESTS)
 foreach ( LITMUS_TEST ${FORALL_LITMUS_TESTS} )
   raja_add_test( NAME test-forall-atomic-litmus-${LITMUS_BACKEND}-${LITMUS_TEST}
                  SOURCES test-forall-atomic-litmus-${LITMUS_TEST}.cpp)
   target_include_directories(test-forall-atomic-litmus-${LITMUS_BACKEND}-${LITMUS_TEST}.exe
                  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
 endforeach()
endif()

unset(FORALL_LITMUS_TESTS)
unset(ENABLE_LITMUS_TESTS)
unset(LITMUS_BACKEND)
