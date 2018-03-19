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
# Set version number
set(RAJA_VERSION_MAJOR 0)
set(RAJA_VERSION_MINOR 6)
set(RAJA_VERSION_PATCHLEVEL 0)

if (RAJA_LOADED AND (NOT RAJA_LOADED STREQUAL "${RAJA_VERSION_MAJOR}.${RAJA_VERSION_MINOR}.${RAJA_VERSION_PATCHLEVEL}"))
  message(FATAL_ERROR "You are mixing RAJA versions. Loaded is ${RAJA_LOADED}, expected ${RAJA_VERSION_MAJOR}.${RAJA_VERSION_MINOR}.${RAJA_VERSION_PATCHLEVEL}")
endif()

if (NOT RAJA_LOADED)
  set (RAJA_LOADED "${RAJA_VERSION_MAJOR}.${RAJA_VERSION_MINOR}.${RAJA_VERSION_PATCHLEVEL}")
  set (RAJA_LOADED ${RAJA_LOADED} PARENT_SCOPE)

  mark_as_advanced(RAJA_LOADED)

  cmake_minimum_required (VERSION 3.3)

  # C is required for googletest to find Threads
  project(RAJA LANGUAGES CXX C
    VERSION ${RAJA_LOADED})

  set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/thirdparty" ${CMAKE_MODULE_PATH})

  # Build options
  set(ENABLE_OPENMP On CACHE Bool "Build OpenMP support")
  set(ENABLE_CUDA Off CACHE Bool "Build CUDA support")
  set(ENABLE_COPY_HEADERS Off CACHE Bool "")
  set(ENABLE_WARNINGS_AS_ERRORS Off CACHE Bool "")

  set(RAJA_CXX_STANDARD_FLAG "default" CACHE STRING "Specific c++ standard flag to use, default attempts to autodetect the highest available")

  option(ENABLE_TBB "Build TBB support" Off)
  option(ENABLE_TARGET_OPENMP "Build OpenMP on target device support" Off)
  option(ENABLE_CLANG_CUDA "Use Clang's native CUDA support" Off)
  set(CUDA_ARCH "sm_35" CACHE STRING "Compute architecture to pass to CUDA builds")
  option(ENABLE_TESTS "Build tests" On)
  option(ENABLE_EXAMPLES "Build simple examples" On)
  option(ENABLE_MODULES "Enable modules in supporting compilers (clang)" On)
  option(ENABLE_WARNINGS "Enable warnings as errors for CI" Off)
  option(ENABLE_DOCUMENTATION "Build RAJA documentation" Off)
  option(ENABLE_COVERAGE "Enable coverage (only supported with GCC)" Off)

  set(TEST_DRIVER "" CACHE STRING "driver used to wrap test commands")

  if (NOT BLT_LOADED) 
  if (NOT EXISTS ${PROJECT_SOURCE_DIR}/blt/SetupBLT.cmake)
    message(FATAL_ERROR "\
    The BLT submodule is not present. \
    If in a git repo run the following command:\n\
    git submodule init && git submodule update")
  endif()

  include(blt/SetupBLT.cmake)
  endif()

  # Setup basic CMake options
  include(cmake/SetupBasics.cmake)
  # Find third-party packages
  include(cmake/SetupPackages.cmake)
  # Setup vendor-specific compiler flags
  include(cmake/SetupCompilers.cmake)
  # Setup internal RAJA configuration options
  include(cmake/SetupRajaConfig.cmake)
  # Macros for building executables and libraries
  include (cmake/RAJAMacros.cmake)

  set (raja_sources
    src/AlignedRangeIndexSetBuilders.cpp
    src/DepGraphNode.cpp
    src/LockFreeIndexSetBuilders.cpp
    src/MemUtils_CUDA.cpp
    src/ThreadUtils_CPU.cpp)

  set (raja_depends)

  if (ENABLE_OPENMP)
    set (raja_depends
      openmp)
  endif()

  if (ENABLE_CUDA)
    set (raja_depends
      ${raja_depends}
      cuda)
  endif ()

  if (ENABLE_CHAI)
    set (raja_depends
      ${raja_depends}
      chai)
  endif ()

  if (ENABLE_TBB)
    set(raja_depends
      ${raja_depends}
      tbb)
  endif ()

  blt_add_library(
    NAME RAJA
    SOURCES ${raja_sources}
    DEPENDS_ON ${raja_depends})

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
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tpl/cub>
    $<INSTALL_INTERFACE:include>)

  install(DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN *.hpp)
  install(DIRECTORY tpl/cub/ DESTINATION include FILES_MATCHING PATTERN *.cuh)

  install(FILES
    ${PROJECT_BINARY_DIR}/include/RAJA/config.hpp
    include/RAJA/module.modulemap
    include/RAJA/module.private.modulemap
    DESTINATION "include/RAJA/")

  if(ENABLE_TESTS)
    add_subdirectory(test)
  endif()

  if(ENABLE_EXAMPLES)
    add_subdirectory(examples)
  endif()

  if (ENABLE_DOCUMENTATION)
    add_subdirectory(docs)
  endif ()

endif()
