###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of experimental reduction types for generating test files.
#
set(REDUCETYPES ReduceSum ReduceMin ReduceMax ReduceMaxLoc ReduceMinLoc)

set(DATATYPES CoreReductionDataTypeList)

#
# List of core reduction types for generating test files.
#
#set(REDUCETYPES ReduceSum ReduceMin ReduceMax ReduceMinLoc ReduceMaxLoc)
set(REDUCETYPES ReduceSum)

set(DATATYPES CoreReductionDataTypeList)


#
# Generate core reduction tests for each enabled RAJA back-end
#
# Note: LAUNCH_BACKENDS is defined in ../CMakeLists.txt
#
foreach( BACKEND ${LAUNCH_BACKENDS} )
  if( NOT (BACKEND STREQUAL "Sycl"))
  foreach( REDUCETYPE ${REDUCETYPES} )
    configure_file( test-launch-basic-reduce.cpp.in
                    test-launch-basic-${REDUCETYPE}-${BACKEND}.cpp )
    raja_add_test( NAME test-launch-basic-${REDUCETYPE}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-launch-basic-${REDUCETYPE}-${BACKEND}.cpp )

    target_include_directories(test-launch-basic-${REDUCETYPE}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
  endif()
endforeach()

unset( DATATYPES )
unset( REDUCETYPES )

#[===[

#
# List of bitwise reduction types for generating test files.
#
set(REDUCETYPES ReduceBitAnd ReduceBitOr)

set(DATATYPES BitwiseReductionDataTypeList)

#
# Generate bitwise reduction tests for each enabled RAJA back-end
#
# Note: LAUNCH_BACKENDS is defined in ../CMakeLists.txt
#
foreach( BACKEND ${LAUNCH_BACKENDS} )
  foreach( REDUCETYPE ${REDUCETYPES} )
    configure_file( test-launch-basic-reduce.cpp.in
                    test-launch-basic-${REDUCETYPE}-${BACKEND}.cpp )
    raja_add_test( NAME test-launch-basic-${REDUCETYPE}-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-launch-basic-${REDUCETYPE}-${BACKEND}.cpp )

    target_include_directories(test-launch-basic-${REDUCETYPE}-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endforeach()

unset( DATATYPES )
unset( REDUCETYPES )

]===]
