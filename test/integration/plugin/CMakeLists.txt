###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
################################################################################

#
# Generate tests for each enabled RAJA back-end.
#
# Note: PLUGIN_BACKENDS is defined in ../CMakeLists.txt
#
foreach( BACKEND ${PLUGIN_BACKENDS} )
  configure_file( test-plugin-forall.cpp.in
                  test-plugin-forall-${BACKEND}.cpp )
  raja_add_test( NAME test-plugin-forall-${BACKEND}
                 SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-plugin-forall-${BACKEND}.cpp
                         plugin_to_test.cpp )

  target_include_directories(test-plugin-forall-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()

foreach( BACKEND ${PLUGIN_BACKENDS} )
  configure_file( test-plugin-kernel.cpp.in
                  test-plugin-kernel-${BACKEND}.cpp )
  raja_add_test( NAME test-plugin-kernel-${BACKEND}
                 SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-plugin-kernel-${BACKEND}.cpp
                         plugin_to_test.cpp )

  target_include_directories(test-plugin-kernel-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()

foreach( BACKEND ${PLUGIN_BACKENDS} )
  configure_file( test-plugin-workgroup.cpp.in
                  test-plugin-workgroup-${BACKEND}.cpp )

  #Some tests are known to fail for Hip, mark those tests (Will not be run in Gitlab CI)
  if(${BACKEND} STREQUAL "Hip")
      raja_add_test( NAME test-plugin-workgroup-Known-Hip-Failure-${BACKEND}
                 SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-plugin-workgroup-${BACKEND}.cpp
                         plugin_to_test.cpp )

      target_include_directories(test-plugin-workgroup-Known-Hip-Failure-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  else()
      raja_add_test( NAME test-plugin-workgroup-${BACKEND}
                 SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-plugin-workgroup-${BACKEND}.cpp
                         plugin_to_test.cpp )

      target_include_directories(test-plugin-workgroup-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endif()
endforeach()

