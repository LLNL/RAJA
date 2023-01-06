###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of indexset test types for generating test files.
#
set(INDEXSETTESTTYPES ResourceIndexSet ResourceIcountIndexSet)

if(RAJA_ENABLE_TARGET_OPENMP)
  if(RAJA_TEST_OPENMP_TARGET_SUBSET)
    list(REMOVE_ITEM FORALL_BACKENDS OpenMPTarget)
  endif()
endif()

#
# Generate tests for each enabled RAJA back-end.
#
# Note: FORALL_BACKENDS is defined in ../CMakeLists.txt
#
foreach( BACKEND ${FORALL_BACKENDS} )
  foreach( INDEXSETTESTTYPE ${INDEXSETTESTTYPES} )
    configure_file( test-forall-resource-indexset.cpp.in
                    test-forall-${INDEXSETTESTTYPE}-${BACKEND}.cpp )
    raja_add_test( NAME test-forall-${INDEXSETTESTTYPE}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-forall-${INDEXSETTESTTYPE}-${BACKEND}.cpp )

    target_include_directories(test-forall-${INDEXSETTESTTYPE}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( INDEXSETTESTTYPES )
