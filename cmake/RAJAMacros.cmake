###############################################################################
# Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
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
  set(singleValueArgs NAME TEST)
  set(multiValueArgs SOURCES DEPENDS_ON)

  cmake_parse_arguments(arg
    "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

  list (APPEND arg_DEPENDS_ON RAJA)

  if (ENABLE_CHAI)
    list (APPEND arg_DEPENDS_ON chai)
  endif ()

  if (ENABLE_OPENMP)
    list (APPEND arg_DEPENDS_ON openmp)
  endif ()

  if (ENABLE_CUDA)
    list (APPEND arg_DEPENDS_ON cuda)
  endif ()

  if (ENABLE_TBB)
    list (APPEND arg_DEPENDS_ON tbb)
  endif ()

  if (${arg_TEST})
    set (_output_dir test)
  else ()
    set (_output_dir bin)
  endif()

  blt_add_executable(
    NAME ${arg_NAME}
    SOURCES ${arg_SOURCES}
    DEPENDS_ON ${arg_DEPENDS_ON}
    OUTPUT_DIR ${_output_dir}
    )
endmacro(raja_add_executable)

macro(raja_add_test)
  set(options )
  set(singleValueArgs NAME)
  set(multiValueArgs SOURCES DEPENDS_ON)

  cmake_parse_arguments(arg
    "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

  list (APPEND arg_DEPENDS_ON gtest ${CMAKE_THREAD_LIBS_INIT})

  raja_add_executable(
    NAME ${arg_NAME}.exe
    SOURCES ${arg_SOURCES}
    DEPENDS_ON ${arg_DEPENDS_ON}
    TEST On)

  blt_add_test(
    NAME ${arg_NAME}
    #COMMAND ${TEST_DRIVER} $<TARGET_FILE:${arg_NAME}>)
    COMMAND ${TEST_DRIVER} ${arg_NAME})
endmacro(raja_add_test)
