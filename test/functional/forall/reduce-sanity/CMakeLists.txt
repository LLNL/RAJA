###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# Function that generates test file and build target for each
# reduction type and each backend
#
function( buildforallreducesanitytest REDUCETYPES )
  foreach( REDUCETYPE ${REDUCETYPES} )
    configure_file( test-forall-reduce-sanity.cpp.in 
                    test-forall-${REDUCETYPE}-sanity-${BACKEND}.cpp )
    raja_add_test( NAME test-forall-${REDUCETYPE}-sanity-${BACKEND}
                   SOURCES test-forall-${REDUCETYPE}-sanity-${BACKEND}.cpp )

    target_include_directories(test-forall-${REDUCETYPE}-sanity-${BACKEND}.exe 
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endfunction()


#
# Variable holding reduction types
#
set(REDUCETYPES ReduceSum ReduceMin ReduceMax ReduceMinLoc ReduceMaxLoc)


#
# Generate tests for each enabled RAJA back-end
#

set( BACKEND Sequential )
set( RESOURCE Host )
  buildforallreducesanitytest( "${REDUCETYPES}" )
unset( RESOURCE )
unset( BACKEND )

if(RAJA_ENABLE_OPENMP)
  set( BACKEND OpenMP )
  set( RESOURCE Host )
    buildforallreducesanitytest( "${REDUCETYPES}" )
  unset( RESOURCE )
  unset( BACKEND )
endif()

if(RAJA_ENABLE_TBB)
  set( BACKEND TBB )
  set( RESOURCE Host )
    buildforallreducesanitytest( "${REDUCETYPES}" )
  unset( RESOURCE )
  unset( BACKEND )
endif()

if(RAJA_ENABLE_CUDA)
  set( BACKEND Cuda )
  set( RESOURCE Cuda )
    buildforallreducesanitytest( "${REDUCETYPES}" )
  unset( RESOURCE )
  unset( BACKEND )
endif()

if(RAJA_ENABLE_HIP)
  set( BACKEND Hip )
  set( RESOURCE Hip )
    buildforallreducesanitytest( "${REDUCETYPES}" )
  unset( RESOURCE )
  unset( BACKEND )
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  set( BACKEND OpenMPTarget )
  set( RESOURCE OpenMPTarget )
    buildforallreducesanitytest( "${REDUCETYPES}" )
  unset( RESOURCE )
  unset( BACKEND )
endif()
