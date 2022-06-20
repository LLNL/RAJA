###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and other RAJA project contributors. See the RAJA/LICENSE file for details.
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

  if (RAJA_ENABLE_OPENMP)
    list (APPEND arg_DEPENDS_ON openmp)
  endif ()

  if (RAJA_ENABLE_CUDA)
    list (APPEND arg_DEPENDS_ON cuda)
  endif ()

  if (RAJA_ENABLE_HIP)
    list (APPEND arg_DEPENDS_ON blt::hip)
    list (APPEND arg_DEPENDS_ON blt::hip_runtime)
  endif ()

  if (RAJA_ENABLE_SYCL)
    list (APPEND arg_DEPENDS_ON sycl)
  endif ()

  if (RAJA_ENABLE_TBB)
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

macro(raja_add_plugin_library)
  set(options )
  set(singleValueArgs NAME SHARED)
  set(multiValueArgs SOURCES DEPENDS_ON)

  cmake_parse_arguments(arg
    "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

  list(APPEND arg_DEPENDS_ON RAJA)

  if (RAJA_ENABLE_OPENMP)
    list (APPEND arg_DEPENDS_ON openmp)
  endif ()

  if (RAJA_ENABLE_CUDA)
    list (APPEND arg_DEPENDS_ON cuda)
  endif ()

  if (RAJA_ENABLE_HIP)
    list (APPEND arg_DEPENDS_ON blt::hip)
    list (APPEND arg_DEPENDS_ON blt::hip_runtime)
  endif ()

  if (RAJA_ENABLE_SYCL)
    list (APPEND arg_DEPENDS_ON sycl)
  endif ()

  if (RAJA_ENABLE_TBB)
    list (APPEND arg_DEPENDS_ON tbb)
  endif ()

  blt_add_library(
    NAME ${arg_NAME}
    SOURCES ${arg_SOURCES}
    DEPENDS_ON ${arg_DEPENDS_ON}
    SHARED ${arg_SHARED}
    )

  #target_include_directories(${arg_NAME}
  #PUBLIC
  #$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  #$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  #$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/tpl/cub>
  #$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/tpl/camp/include>
  #$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/tpl/rocPRIM/rocprim/include>
  #$<INSTALL_INTERFACE:include>)

endmacro(raja_add_plugin_library)

# Allows strings embedded in test files to used to process ctest results.
# Only works for new testing framework/structure (no effect on old tests).
# Borrowed heavily from CAMP.
function(raja_set_failtest TESTNAME)
  set(test_name ${TESTNAME})

  # Chopping off backend from test name
  string(REGEX REPLACE "\-Sequential|\-OpenMP|\-OpenMPTarget|\-TBB|\-CUDA|\-HIP" "" test_nobackend ${test_name})

  # Finding test source code
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/${test_nobackend}.hpp")
    list(APPEND TEST_LIST "${CMAKE_CURRENT_SOURCE_DIR}/tests/${test_nobackend}.hpp")
    list(REMOVE_DUPLICATES TEST_LIST)
  endif()

  list(GET TEST_LIST 0 source_file)

  if(EXISTS ${source_file})
    set(test_regex  ".*(WILL_FAIL|PASS_REGEX|FAIL_REGEX):?[ ]*(.*)[ ]*")

    file(STRINGS ${source_file} test_lines REGEX "${test_regex}")

    # Search test source code for fail string
    foreach(line ${test_lines})
      if(NOT line MATCHES "${test_regex}")
        continue()
      endif()

      if(CMAKE_MATCH_1 STREQUAL "WILL_FAIL")
        set_property( TARGET ${test_name}.exe   # TARGET more conformant to BLT
                      APPEND PROPERTY WILL_FAIL )
      elseif(CMAKE_MATCH_1 STREQUAL "PASS_REGEX")
        set_property( TARGET ${test_name}.exe
                      APPEND PROPERTY PASS_REGULAR_EXPRESSION "${CMAKE_MATCH_2}")
      elseif(CMAKE_MATCH_1 STREQUAL "FAIL_REGEX")
        set_property( TARGET ${test_name}.exe
                      APPEND PROPERTY FAIL_REGULAR_EXPRESSION "${CMAKE_MATCH_2}")
      endif()
    endforeach()
  endif()
endfunction()

macro(raja_add_test)
  set(options )
  set(singleValueArgs NAME)
  set(multiValueArgs SOURCES DEPENDS_ON)

  cmake_parse_arguments(arg
    "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

  list (APPEND arg_DEPENDS_ON gtest ${CMAKE_THREAD_LIBS_INIT})

  set(original_test_name ${arg_NAME})

  raja_add_executable(
    NAME ${arg_NAME}.exe
    SOURCES ${arg_SOURCES}
    DEPENDS_ON ${arg_DEPENDS_ON}
    TEST On)

  blt_add_test(
    NAME ${arg_NAME}
    #COMMAND ${TEST_DRIVER} $<TARGET_FILE:${arg_NAME}>)
    COMMAND ${TEST_DRIVER} ${arg_NAME})

  raja_set_failtest(${original_test_name})
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
