###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# Generate tests for each enabled RAJA back-end.
#
# Note: GRAPH_FORALL_BACKENDS is defined in ../CMakeLists.txt
#
foreach(GRAPHBACKEND ${GRAPH_BACKENDS})
  configure_file( test-graph-MixedNodes.cpp.in
                  test-graph-${GRAPHBACKEND}-MixedNodes.cpp )
  raja_add_test( NAME test-graph-${GRAPHBACKEND}-MixedNodes
                 SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-graph-${GRAPHBACKEND}-MixedNodes.cpp )

  target_include_directories(test-graph-${GRAPHBACKEND}-MixedNodes.exe
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()
