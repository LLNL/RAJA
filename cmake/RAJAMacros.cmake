###############################################################################
# Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-689114
#
# All rights reserved.
#
# This file is part of RAJA.
#
# For details about use and distribution, please read RAJA/LICENSE.
#
###############################################################################

macro(raja_add_executable)
  set(options )
  set(singleValueArgs NAME)
  set(multiValueArgs SOURCES DEPENDS_ON)

  cmake_parse_arguments(arg
    "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

  if (RAJA_ENABLE_CHAI)
    list (APPEND arg_DEPENDS_ON chai)
  endif ()

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
      target_link_libraries(${arg_NAME} PUBLIC RAJA ${arg_DEPENDS_ON})
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

  if (RAJA_ENABLE_CHAI)
    list (APPEND arg_DEPENDS_ON chai)
  endif ()

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

endmacro(raja_add_library)

macro(raja_add_test)
  set(options )
  set(singleValueArgs NAME)
  set(multiValueArgs SOURCES DEPENDS_ON)

  cmake_parse_arguments(arg
    "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

  list (APPEND arg_DEPENDS_ON gtest gtest_main ${CMAKE_THREAD_LIBS_INIT})

  raja_add_executable(
    NAME ${arg_NAME}.exe
    SOURCES ${arg_SOURCES}
    DEPENDS_ON ${arg_DEPENDS_ON})

  add_test(NAME ${arg_NAME}
    COMMAND ${TEST_DRIVER} $<TARGET_FILE:${arg_NAME}>)
endmacro(raja_add_test)
