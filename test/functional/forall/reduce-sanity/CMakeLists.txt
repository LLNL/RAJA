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
function( buildforallreducesanitytest reducetype backend )
  configure_file( test-forall-reduce-sanity.cpp.in 
                  test-forall-${reducetype}-sanity-${backend}.cpp )
  raja_add_test( NAME test-forall-${reducetype}-sanity-${backend}
                 SOURCES test-forall-${reducetype}-sanity-${backend}.cpp )

  target_include_directories(test-forall-${reducetype}-sanity-${backend}.exe 
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endfunction()

#
# Variable holding reduction types
#
set(REDUCETYPES ReduceSum ReduceMin ReduceMax ReduceMinLoc ReduceMaxLoc)


#
# Generate tests for each enabled RAJA back-end
#
set( BACKEND Sequential )
foreach( REDUCETYPE ${REDUCETYPES} )
  buildforallreducesanitytest( ${REDUCETYPE} ${BACKEND} )
endforeach()
unset( BACKEND )

if(RAJA_ENABLE_OPENMP)
  set( BACKEND OpenMP )
  foreach( REDUCETYPE ${REDUCETYPES} )
    buildforallreducesanitytest( ${REDUCETYPE} ${BACKEND} )
  endforeach()
  unset( BACKEND )
endif()

if(RAJA_ENABLE_TBB)
  set( BACKEND TBB )
  foreach( REDUCETYPE ${REDUCETYPES} )
    buildforallreducesanitytest( ${REDUCETYPE} ${BACKEND} )
  endforeach()
  unset( BACKEND )
endif()

if(RAJA_ENABLE_CUDA)
  set( BACKEND Cuda )
  foreach( REDUCETYPE ${REDUCETYPES} )
    buildforallreducesanitytest( ${REDUCETYPE} ${BACKEND} )
  endforeach()
  unset( BACKEND )
endif()

if(RAJA_ENABLE_HIP)
  set( BACKEND Hip )
  foreach( REDUCETYPE ${REDUCETYPES} )
    buildforallreducesanitytest( ${REDUCETYPE} ${BACKEND} )
  endforeach()
  unset( BACKEND )
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  set( BACKEND OpenMPTarget )
  foreach( REDUCETYPE ${REDUCETYPES} )
    buildforallreducesanitytest( ${REDUCETYPE} ${BACKEND} )
  endforeach()
  unset( BACKEND )
endif()
