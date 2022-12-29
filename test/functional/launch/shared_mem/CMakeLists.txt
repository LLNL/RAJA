###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of segment types for generating test files.
#
# StaticMem
set(SHARED_MEM_TYPES DynamicMem)


#
# Generate tests for each enabled RAJA back-end.
#
#

foreach( BACKEND ${LAUNCH_BACKENDS} )
  foreach( SHARED_MEM_TYPES ${SHARED_MEM_TYPES} )
    configure_file( test-launch-shared-mem.cpp.in
                    test-launch-shared-mem-${SHARED_MEM_TYPES}-${BACKEND}.cpp )
    raja_add_test( NAME test-launch-shared-mem-${SHARED_MEM_TYPES}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-launch-shared-mem-${SHARED_MEM_TYPES}-${BACKEND}.cpp )

    target_include_directories(test-launch-shared-mem-${SHARED_MEM_TYPES}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( SHARED_MEM_TYPES )
