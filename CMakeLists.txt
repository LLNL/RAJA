###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

cmake_policy(SET CMP0042 NEW)
cmake_policy(SET CMP0048 NEW)

if (APPLE)
 cmake_policy(SET CMP0025 NEW)
endif()

include(CMakeDependentOption)

# Set version number
set(RAJA_VERSION_MAJOR 0)
set(RAJA_VERSION_MINOR 13)
set(RAJA_VERSION_PATCHLEVEL 0)

if (RAJA_LOADED AND (NOT RAJA_LOADED STREQUAL "${RAJA_VERSION_MAJOR}.${RAJA_VERSION_MINOR}.${RAJA_VERSION_PATCHLEVEL}"))
  message(FATAL_ERROR "You are mixing RAJA versions. Loaded is ${RAJA_LOADED}, expected ${RAJA_VERSION_MAJOR}.${RAJA_VERSION_MINOR}.${RAJA_VERSION_PATCHLEVEL}")
endif()

if (RAJA_LOADED)
  return() # Stop processing file, avoids nesting the whole file
endif()
set (RAJA_LOADED "${RAJA_VERSION_MAJOR}.${RAJA_VERSION_MINOR}.${RAJA_VERSION_PATCHLEVEL}")

# Promote RAJA_LOADED to PARENT_SCOPE if it exists, which is only if we are bringing
# in RAJA as a subproject to a larger CMake project
get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)
  set (RAJA_LOADED ${RAJA_LOADED} PARENT_SCOPE)
endif()

mark_as_advanced(RAJA_LOADED)

# C is required for googletest to find Threads
project(RAJA LANGUAGES CXX C
  VERSION ${RAJA_LOADED})

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/thirdparty" ${CMAKE_MODULE_PATH})

# Build options
set(ENABLE_OPENMP On CACHE BOOL "Build OpenMP support")
set(ENABLE_CUDA Off CACHE BOOL "Build CUDA support")
set(ENABLE_COPY_HEADERS Off CACHE BOOL "")
set(ENABLE_WARNINGS_AS_ERRORS Off CACHE BOOL "")
set(ENABLE_GTEST_DEATH_TESTS On CACHE BOOL "Enable tests asserting failure.")

set(RAJA_CXX_STANDARD_FLAG "default" CACHE STRING "Specific c++ standard flag to use, default attempts to autodetect the highest available")

## NOTE: CMake-dependent options are placed AFTER BLT is loaded so they 
##       work as intended. BLT has variables defined for these and we use
##       the same names with 'RAJA_' prepended to them.

option(ENABLE_TBB "Build TBB support" Off)
option(ENABLE_TARGET_OPENMP "Build OpenMP on target device support" Off)
option(ENABLE_CLANG_CUDA "Use Clang's native CUDA support" Off)
option(ENABLE_EXTERNAL_CUB "Use an external cub" Off)
option(ENABLE_EXTERNAL_ROCPRIM "Use an external rocPRIM" Off)

option(ENABLE_REPRODUCERS "Build issue reproducers" Off)

option(RAJA_ENABLE_EXERCISES "Build exercises " On)
option(ENABLE_MODULES "Enable modules in supporting compilers (clang)" On)
option(ENABLE_WARNINGS "Enable warnings as errors for CI" Off)
option(ENABLE_DOCUMENTATION "Build RAJA documentation" Off)
option(ENABLE_COVERAGE "Enable coverage (only supported with GCC)" Off)
option(ENABLE_FORCEINLINE_RECURSIVE "Enable Forceinline recursive (only supported with Intel compilers)" On)

option(RAJA_DEPRECATED_TESTS "Test deprecated features" Off)
option(RAJA_ENABLE_BOUNDS_CHECK "Enable bounds checking in RAJA::Views/Layouts" Off)
option(RAJA_TEST_EXHAUSTIVE "Build RAJA exhaustive tests" Off)
option(RAJA_TEST_OPENMP_TARGET_SUBSET "Build subset of RAJA OpenMP target tests when it is enabled" On)
option(RAJA_ENABLE_RUNTIME_PLUGINS "Enable support for loading plugins at runtime" Off)

set(TEST_DRIVER "" CACHE STRING "driver used to wrap test commands")

set(BLT_EXPORT_THIRDPARTY ON CACHE BOOL "")

cmake_minimum_required(VERSION 3.9)

if (ENABLE_CUDA)
  if (DEFINED CUDA_ARCH)
    if (CUDA_ARCH MATCHES "^sm_*")
      if ("${CUDA_ARCH}" STRLESS "sm_35")
        message( FATAL_ERROR "RAJA requires minimum CUDA compute architecture of sm_35")
      endif()
    endif()
    if (CUDA_ARCH MATCHES "^compute_*")
      if ("${CUDA_ARCH}" STRLESS "compute_35")
        message( FATAL_ERROR "RAJA requires minimum CUDA compute architecture of compute_35")
      endif()
    endif()
  else()
    message(STATUS "CUDA compute architecture set to RAJA default sm_35 since it was not specified")
    set(CUDA_ARCH "sm_35" CACHE STRING "Set CUDA_ARCH to RAJA minimum supported" FORCE)
  endif()
  if ( (CMAKE_CXX_COMPILER_ID MATCHES GNU) AND (CMAKE_SYSTEM_PROCESSOR MATCHES ppc64le) )
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 8.0)
      set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -mno-float128")
    endif ()
  endif ()
endif()

# Detect C++ standard and add appropriate flag _before_ loading BLT
set(COMPILERS_KNOWN_TO_CMAKE33 AppleClang Clang GNU MSVC)

include(CheckCXXCompilerFlag)
if(RAJA_CXX_STANDARD_FLAG MATCHES default)
  if("cxx_std_17" IN_LIST CMAKE_CXX_KNOWN_FEATURES)
    #TODO set BLT_CXX_STANDARD
    set(CMAKE_CXX_STANDARD 17)
  elseif("cxx_std_14" IN_LIST CMAKE_CXX_KNOWN_FEATURES)
    set(CMAKE_CXX_STANDARD 14)
  elseif("${CMAKE_CXX_COMPILER_ID}" IN_LIST COMPILERS_KNOWN_TO_CMAKE33)
    set(CMAKE_CXX_STANDARD 14)
  else() #cmake has no idea what to do, do it ourselves...
    foreach(flag_var "-std=c++17" "-std=c++1z" "-std=c++14" "-std=c++1y" "-std=c++11")
      CHECK_CXX_COMPILER_FLAG(${flag_var} COMPILER_SUPPORTS_${flag_var})
      if(COMPILER_SUPPORTS_${flag_var})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag_var}")
        break()
      endif()
    endforeach(flag_var)
  endif()
else(RAJA_CXX_STANDARD_FLAG MATCHES default)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${RAJA_CXX_STANDARD_FLAG}")
  message("Using C++ standard flag: ${RAJA_CXX_STANDARD_FLAG}")
endif(RAJA_CXX_STANDARD_FLAG MATCHES default)

set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT BLT_LOADED)
  if (DEFINED BLT_SOURCE_DIR)
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
      message(FATAL_ERROR "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake")
    endif()
  else ()
    set (BLT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/blt CACHE PATH "")

    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
      message(FATAL_ERROR "\
      The BLT submodule is not present. \
      If in git repository run the following two commands:\n \
      git submodule init\n \
      git submodule update")
    endif ()
  endif ()

  include(${BLT_SOURCE_DIR}/SetupBLT.cmake)
endif()

##
## Here are the CMake dependent options...
##
cmake_dependent_option(RAJA_ENABLE_TESTS "Build tests" On "ENABLE_TESTS" Off)
cmake_dependent_option(RAJA_ENABLE_EXAMPLES "Build simple examples" On "ENABLE_EXAMPLES" off)
cmake_dependent_option(RAJA_ENABLE_BENCHMARKS "Build benchmarks" On "ENABLE_BENCHMARKS" Off)


# Setup basic CMake options
include(cmake/SetupBasics.cmake)
# Find third-party packages
include(cmake/SetupPackages.cmake)
# Setup vendor-specific compiler flags
include(cmake/SetupCompilers.cmake)

# Macros for building executables and libraries
include (cmake/RAJAMacros.cmake)

set (raja_sources
  src/AlignedRangeIndexSetBuilders.cpp
  src/DepGraphNode.cpp
  src/LockFreeIndexSetBuilders.cpp
  src/MemUtils_CUDA.cpp
  src/MemUtils_HIP.cpp
  src/PluginStrategy.cpp)

if (RAJA_ENABLE_RUNTIME_PLUGINS)
  set (raja_sources
    ${raja_sources}
    src/RuntimePluginLoader.cpp
    src/KokkosPluginLoader.cpp)
endif ()

set (raja_depends)

if (ENABLE_OPENMP)
  set (raja_depends
    openmp)
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 17)
  message(WARNING "RAJA::simd_exec support requires Intel-17 or greater")
endif()

if (ENABLE_CUDA)
  set(raja_depends
    ${raja_depends}
    cuda)
endif ()

if (ENABLE_EXTERNAL_CUB)
  set(raja_depends
    ${raja_depends}
    cub)
endif ()

if (ENABLE_HIP)
  set(raja_depends
    ${raja_depends}
    hip)
endif ()

if (ENABLE_EXTERNAL_ROCPRIM)
  set(raja_depends
    ${raja_depends}
    rocPRIM)
endif ()

if (ENABLE_TBB)
  set(raja_depends
    ${raja_depends}
    tbb)
endif ()

if (NOT TARGET camp)
  set(EXTERNAL_CAMP_SOURCE_DIR "" CACHE FILEPATH "build with a specific external
camp source repository")
  if (EXTERNAL_CAMP_SOURCE_DIR)
    message(STATUS "Using external source CAMP from: " ${EXTERNAL_CAMP_SOURCE_DIR})
    add_subdirectory(${EXTERNAL_CAMP_SOURCE_DIR}
                     ${CMAKE_CURRENT_BINARY_DIR}/tpl/camp)
  else (EXTERNAL_CAMP_SOURCE_DIR)
    find_package(camp QUIET)
    if (NOT camp_FOUND)
      message(STATUS "Using RAJA CAMP submodule.")
      add_subdirectory(tpl/camp)
    else (NOT camp_FOUND)
      message(STATUS "Using installed CAMP from:  ${camp_INSTALL_PREFIX}")
      # Save the path so it can be used in raja-config.cmake
      set(EXTERNAL_CAMP_INSTALL_DIR ${camp_INSTALL_PREFIX} CACHE PATH "")
    endif(NOT camp_FOUND)
  endif (EXTERNAL_CAMP_SOURCE_DIR)
endif (NOT TARGET camp)

set (raja_defines)

if (COMPILER_FAMILY_IS_MSVC AND NOT BUILD_SHARED_LIBS)
  set (raja_defines
    ${raja_defines}
    RAJA_WIN_STATIC_BUILD)
endif ()

blt_add_library(
  NAME RAJA
  SOURCES ${raja_sources}
  DEPENDS_ON ${raja_depends} camp ${CMAKE_DL_LIBS}
  DEFINES ${raja_defines})


install(TARGETS RAJA
  EXPORT RAJA
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib
  )

install(EXPORT RAJA DESTINATION share/raja/cmake/)

target_include_directories(RAJA
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)
if (ENABLE_CUDA AND NOT ENABLE_EXTERNAL_CUB)
  target_include_directories(RAJA SYSTEM
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tpl/cub>)
endif()
if (ENABLE_HIP AND NOT ENABLE_EXTERNAL_ROCPRIM)
  target_include_directories(RAJA SYSTEM
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tpl/rocPRIM/rocprim/include>)
endif()

install(DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN *.hpp)
if (ENABLE_CUDA AND NOT ENABLE_EXTERNAL_CUB)
  install(DIRECTORY tpl/cub/cub/ DESTINATION include/cub FILES_MATCHING PATTERN *.cuh)
endif()
if (ENABLE_HIP AND NOT ENABLE_EXTERNAL_ROCPRIM)
  install(DIRECTORY tpl/rocPRIM/rocprim/include/ DESTINATION include FILES_MATCHING PATTERN *.hpp)
endif()

install(FILES
  ${PROJECT_BINARY_DIR}/include/RAJA/config.hpp
  include/RAJA/module.modulemap
  include/RAJA/module.private.modulemap
  DESTINATION "include/RAJA/")


# Setup internal RAJA configuration options
include(cmake/SetupRajaConfig.cmake)

if(RAJA_ENABLE_TESTS)
  add_subdirectory(test)
endif()

if(ENABLE_REPRODUCERS)
  add_subdirectory(reproducers)
endif()

if(RAJA_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

if(RAJA_ENABLE_EXERCISES)
  add_subdirectory(exercises)
endif()

if (ENABLE_DOCUMENTATION)
  add_subdirectory(docs)
endif ()

if (RAJA_ENABLE_BENCHMARKS)
  add_subdirectory(benchmark)
endif ()
