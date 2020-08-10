###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of reduction types for generating test files.
#
set(REDUCETYPES ReduceBitAnd ReduceBitOr)


#
# Generate tests for each enabled RAJA back-end
#
# Note: FORALL_BACKENDS is defined in ../CMakeLists.txt
#
foreach( BACKEND ${FORALL_BACKENDS} )
  foreach( REDUCETYPE ${REDUCETYPES} )
    configure_file( test-forall-bitwise-reduce.cpp.in
                    test-forall-bitwise-${REDUCETYPE}-${BACKEND}.cpp )
    raja_add_test( NAME test-forall-bitwise-${REDUCETYPE}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-forall-bitwise-${REDUCETYPE}-${BACKEND}.cpp )

    target_include_directories(test-forall-bitwise-${REDUCETYPE}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( REDUCETYPES )
