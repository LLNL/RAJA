###############################################################################
# Copyright (c) 2016, Lawrence Livermore National Security, LLC.
#    
# Produced at the Lawrence Livermore National Laboratory
#    
# LLNL-CODE-689114
# 
# All rights reserved.
#  
# This file is part of RAJA.
#
# For additional details, please also read RAJA/LICENSE.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the disclaimer below.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################

macro(raja_add_executable)

  set(options )
  set(singleValueArgs NAME)
  set(multiValueArgs SOURCES DEPENDS_ON)

  cmake_parse_arguments(arg
    "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

  if (RAJA_ENABLE_CUDA) 
    if (RAJA_ENABLE_CLANG_CUDA) 
      add_executable(${arg_NAME} ${arg_SOURCES})
      target_compile_options(${arg_NAME} PRIVATE 
        -x cuda --cuda-gpu-arch=${RAJA_CUDA_ARCH} --cuda-path=${CUDA_TOOLKIT_ROOT_DIR})
      target_include_directories(${arg_NAME} 
        PUBLIC ${EXPT_CUDA_INCLUDE_LOCATION})
      target_link_libraries(${arg_NAME} ${CUDA_LIBRARIES} RAJA ${arg_DEPENDS_ON})
    else ()
      set_source_files_properties(
        ${arg_SOURCES}
        PROPERTIES
        CUDA_SOURCE_PROPERTY_FORMAT OBJ)
      cuda_add_executable(${arg_NAME} ${arg_SOURCES})
      target_link_libraries(${arg_NAME} RAJA ${arg_DEPENDS_ON})
    endif()
  else ()
    add_executable(${arg_NAME} ${arg_SOURCES})
    target_link_libraries(${arg_NAME} RAJA ${arg_DEPENDS_ON})
  endif()
endmacro(raja_add_executable)

macro(raja_add_library)
  set(options )
  set(singleValueArgs NAME)
  set(multiValueArgs SOURCES DEPENDS_ON)

  cmake_parse_arguments(arg
    "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

  if (RAJA_ENABLE_CUDA)
    if (RAJA_ENABLE_CLANG_CUDA)

      add_library(${arg_NAME} ${arg_SOURCES})
      target_compile_options(${arg_NAME} PRIVATE 
        -x cuda --cuda-gpu-arch=${RAJA_CUDA_ARCH} --cuda-path=${CUDA_TOOLKIT_ROOT_DIR})
      target_include_directories(${arg_NAME} 
        PUBLIC ${EXPT_CUDA_INCLUDE_LOCATION})
      target_link_libraries(${arg_NAME} ${CUDA_LIBRARIES})

    else ()
      set_source_files_properties(
        ${arg_SOURCES}
        PROPERTIES
        CUDA_SOURCE_PROPERTY_FORMAT OBJ)

      cuda_add_library(${arg_NAME} ${arg_SOURCES})
    endif ()
  else ()
    add_library(${arg_NAME} ${arg_SOURCES})
  endif ()

  if (NOT (CMAKE_CXX_COMPILER_ID MATCHES Intel OR RAJA_ENABLE_CLANG_CUDA) )
      set_target_properties(${arg_NAME}
      PROPERTIES
      CXX_STANDARD 11
      CXX_STANDARD_REQUIRED YES)
  endif()
endmacro(raja_add_library)

macro(raja_add_test)
  set(options )
  set(singleValueArgs NAME)
  set(multiValueArgs SOURCES DEPENDS_ON)

  cmake_parse_arguments(arg
    "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

  raja_add_executable(
    NAME ${arg_NAME}.exe 
    SOURCES ${arg_SOURCES} 
    DEPENDS_ON ${arg_DEPENDS_ON})

  add_test(NAME ${arg_NAME} 
    COMMAND ${TEST_DRIVER} $<TARGET_FILE:${arg_NAME}>)
endmacro(raja_add_test)
