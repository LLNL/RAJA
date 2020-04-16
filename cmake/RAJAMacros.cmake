###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and other RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
################################################################################

macro(raja_add_executable)
  set(options )
  set(singleValueArgs NAME TEST REPRODUCER BENCHMARK)
  set(multiValueArgs SOURCES DEPENDS_ON)

  cmake_parse_arguments(arg
    "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

  list (APPEND arg_DEPENDS_ON RAJA)

  if (ENABLE_OPENMP)
    list (APPEND arg_DEPENDS_ON openmp)
  endif ()

  if (ENABLE_CUDA)
    list (APPEND arg_DEPENDS_ON cuda)
  endif ()

  if (ENABLE_HIP)
    list (APPEND arg_DEPENDS_ON hip)
  endif ()

  if (ENABLE_TBB)
    list (APPEND arg_DEPENDS_ON tbb)
  endif ()

  if (${arg_TEST})
    set (_output_dir ${CMAKE_BINARY_DIR}/test)
  elseif (${arg_REPRODUCER})
    set (_output_dir ${CMAKE_BINARY_DIR}/reproducers)
  elseif (${arg_BENCHMARK})
    set (_output_dir ${CMAKE_BINARY_DIR}/benchmark)
  else ()
    set (_output_dir ${CMAKE_BINARY_DIR}/bin)
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

macro(raja_add_reproducer)
  set(options )
  set(singleValueArgs NAME)
  set(multiValueArgs SOURCES DEPENDS_ON)

  cmake_parse_arguments(arg
    "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

  raja_add_executable(
    NAME ${arg_NAME}.exe
    SOURCES ${arg_SOURCES}
    DEPENDS_ON ${arg_DEPENDS_ON}
    REPRODUCER On)
endmacro(raja_add_reproducer)

macro(raja_add_benchmark)
  set(options )
  set(singleValueArgs NAME)
  set(multiValueArgs SOURCES DEPENDS_ON)

  cmake_parse_arguments(arg
    "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

  list (APPEND arg_DEPENDS_ON gbenchmark)

  raja_add_executable(
    NAME ${arg_NAME}.exe
    SOURCES ${arg_SOURCES}
    DEPENDS_ON ${arg_DEPENDS_ON}
    BENCHMARK On)

  blt_add_benchmark(
    NAME ${arg_NAME}
    COMMAND ${TEST_DRIVER} ${arg_NAME})
endmacro(raja_add_benchmark)
