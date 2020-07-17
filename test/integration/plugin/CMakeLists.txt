###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
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
  configure_file( test-plugin.cpp.in
                  test-plugin-${BACKEND}.cpp )
  raja_add_test( NAME test-plugin-${BACKEND}
                 SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-plugin-${BACKEND}.cpp
                         plugin_to_test.cpp )

  target_include_directories(test-plugin-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()

