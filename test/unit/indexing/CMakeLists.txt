###############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(INDEXING_BACKENDS "")

if(RAJA_ENABLE_CUDA)
  list(APPEND INDEXING_BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND INDEXING_BACKENDS Hip)
endif()

#
# Generate tests for each enabled RAJA back-end.
#
foreach( INDEXING_BACKEND ${INDEXING_BACKENDS} )
  configure_file( test-indexing-global.cpp.in
                  test-indexing-global-${INDEXING_BACKEND}.cpp )
  raja_add_test( NAME test-indexing-global-${INDEXING_BACKEND}
                 SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-indexing-global-${INDEXING_BACKEND}.cpp )

  target_include_directories(test-indexing-global-${INDEXING_BACKEND}.exe
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()
