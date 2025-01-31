###############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of core reduction types for generating test files.
#
set(REDUCETYPES Sum Min Max BitAnd BitOr)

#
# Do not create tests for OpenMP Target, support not currently implemented
#
if(RAJA_ENABLE_TARGET_OPENMP)
  list(REMOVE_ITEM FORALL_BACKENDS OpenMPTarget)
endif()

#
# Do not create tests for SYCL, support not currently implemented
#
if(RAJA_ENABLE_SYCL)
  list(REMOVE_ITEM FORALL_BACKENDS Sycl)
endif()

#
# Generate core reduction tests for each enabled RAJA back-end
#
# Note: FORALL_BACKENDS is defined in ../CMakeLists.txt
#

foreach( BACKEND ${FORALL_BACKENDS} )
  foreach( REDUCETYPE ${REDUCETYPES} )
    configure_file( test-forall-basic-multi-reduce.cpp.in
                    test-forall-basic-MultiReduce${REDUCETYPE}-${BACKEND}.cpp )
    raja_add_test( NAME test-forall-basic-MultiReduce${REDUCETYPE}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-forall-basic-MultiReduce${REDUCETYPE}-${BACKEND}.cpp )

    target_include_directories(test-forall-basic-MultiReduce${REDUCETYPE}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( REDUCETYPES )
