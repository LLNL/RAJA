###############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of dimensions for generating test files.
#
set(DIMENSIONS 1D 2D 3D)

##
## Enable OpenMP Target tests when support for Combining Adapter is fixed
##
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
  foreach( DIMENSION ${DIMENSIONS} )
    configure_file( test-forall-CombiningAdapter.cpp.in
                    test-forall-CombiningAdapter-${DIMENSION}-${BACKEND}.cpp )
    raja_add_test( NAME test-forall-CombiningAdapter-${DIMENSION}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-forall-CombiningAdapter-${DIMENSION}-${BACKEND}.cpp )

    target_include_directories(test-forall-CombiningAdapter-${DIMENSION}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()
