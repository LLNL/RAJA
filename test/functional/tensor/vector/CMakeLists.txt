###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################


set(TENSOR_VECTOR_TESTS
      CtorGetSet
			MinMaxSumDot
    	FmaFms
    	ForallVectorRef1d
     	ForallVectorRef2d
	 )
				

#
# Generate tensor register tests for each element type, and each register policy
#
foreach( TENSOR_ELEMENT_TYPE ${TENSOR_ELEMENT_TYPES} )
	foreach( TENSOR_VECTOR_TEST ${TENSOR_VECTOR_TESTS} )
	
		set(TEST_NAME test-tensor-vector-${TENSOR_ELEMENT_TYPE}-${TENSOR_VECTOR_TEST})
	
		configure_file( test-tensor-vector.cpp.in  ${TEST_NAME}.cpp )
		
		raja_add_test( NAME ${TEST_NAME} SOURCES ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}.cpp )

		target_include_directories(${TEST_NAME}.exe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)

		unset( TEST_NAME )
		
	endforeach()
endforeach()

unset( TENSOR_VECTOR_TESTS )
