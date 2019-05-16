###############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
###############################################################################

cmake_policy(SET CMP0048 NEW)

# Set version number
set(RAJA_VERSION_MAJOR 0)
set(RAJA_VERSION_MINOR 8)
set(RAJA_VERSION_PATCHLEVEL 0)

if (RAJA_LOADED AND (NOT RAJA_LOADED STREQUAL "${RAJA_VERSION_MAJOR}.${RAJA_VERSION_MINOR}.${RAJA_VERSION_PATCHLEVEL}"))
  message(FATAL_ERROR "You are mixing RAJA versions. Loaded is ${RAJA_LOADED}, expected ${RAJA_VERSION_MAJOR}.${RAJA_VERSION_MINOR}.${RAJA_VERSION_PATCHLEVEL}")
endif()

if (NOT RAJA_LOADED)
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
  set(ENABLE_OPENMP On CACHE Bool "Build OpenMP support")
  set(ENABLE_CUDA Off CACHE Bool "Build CUDA support")
  set(ENABLE_COPY_HEADERS Off CACHE Bool "")
  set(ENABLE_WARNINGS_AS_ERRORS Off CACHE Bool "")

  set(RAJA_CXX_STANDARD_FLAG "default" CACHE STRING "Specific c++ standard flag to use, default attempts to autodetect the highest available")

  option(ENABLE_TBB "Build TBB support" Off)
  option(ENABLE_CHAI "Build CHAI support" Off)
  option(ENABLE_TARGET_OPENMP "Build OpenMP on target device support" Off)
  option(ENABLE_CLANG_CUDA "Use Clang's native CUDA support" Off)
  set(CUDA_ARCH "sm_35" CACHE STRING "Compute architecture to pass to CUDA builds")
  option(ENABLE_EXTERNAL_CUB "Use an external cub for scans" Off)
  option(ENABLE_TESTS "Build tests" On)
  option(ENABLE_REPRODUCERS "Build issue reproducers" Off)
  option(ENABLE_EXAMPLES "Build simple examples" On)
  option(ENABLE_EXERCISES "Build exercises " On)
  option(ENABLE_MODULES "Enable modules in supporting compilers (clang)" On)
  option(ENABLE_WARNINGS "Enable warnings as errors for CI" Off)
  option(ENABLE_DOCUMENTATION "Build RAJA documentation" Off)
  option(ENABLE_COVERAGE "Enable coverage (only supported with GCC)" Off)
  option(ENABLE_FORCEINLINE_RECURSIVE "Enable Forceinline recursive (only supported with Intel compilers)" On)
  option(ENABLE_BENCHMARKS "Build benchmarks" Off)
  option(RAJA_DEPRECATED_TESTS "Test deprecated features" Off)

  set(TEST_DRIVER "" CACHE STRING "driver used to wrap test commands")

if (ENABLE_CUDA)
  cmake_minimum_required(VERSION 3.9)
else ()
  cmake_minimum_required(VERSION 3.8.2)
endif ()

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
    src/MemUtils_CUDA.cpp)

  set (raja_depends)

  if (ENABLE_OPENMP)
    set (raja_depends
      openmp)
  endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 17)
     message(WARNING "RAJA::simd_exec support requires Intel-17 or greater")
  endif()

  if (ENABLE_CUDA)
    set (raja_depends
      ${raja_depends}
      cuda)
  endif ()

  if (ENABLE_CUDA)
    if(ENABLE_EXTERNAL_CUB)
      find_package(CUB)
        if (CUB_FOUND)
          blt_register_library(
            NAME cub
            INCLUDES ${CUB_INCLUDE_DIRS})
            set(raja_depends
                ${raja_depends}
                cub)
          else()
            message(WARNING "External CUB not found.")
            set(ENABLE_EXTERNAL_CUB Off)
         endif()
      endif ()
  endif ()

  if (ENABLE_CHAI)
    set (raja_depends
      ${raja_depends}
      chai
      umpire)
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

  if(ENABLE_EXTERNAL_CUB)
    install(DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN *.hpp)
  else()
    install(DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN *.hpp)
    install(DIRECTORY tpl/cub/ DESTINATION include FILES_MATCHING PATTERN *.cuh)
  endif()

  install(FILES
    ${PROJECT_BINARY_DIR}/include/RAJA/config.hpp
    include/RAJA/module.modulemap
    include/RAJA/module.private.modulemap
    DESTINATION "include/RAJA/")

  if(ENABLE_TESTS)
    add_subdirectory(test)
  endif()

  if(ENABLE_REPRODUCERS)
    add_subdirectory(reproducers)
  endif()

  if(ENABLE_EXAMPLES)
    add_subdirectory(examples)
  endif()

  if(ENABLE_EXERCISES)
    add_subdirectory(exercises)
  endif()

  if (ENABLE_DOCUMENTATION)
    add_subdirectory(docs)
  endif ()

  if (ENABLE_BENCHMARKS)
    add_subdirectory(benchmark)
  endif ()

endif()
