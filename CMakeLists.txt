cmake_minimum_required (VERSION 3.1)

project(RAJA LANGUAGES CXX)

# Set version number
set(RAJA_VERSION_MAJOR 1)
set(RAJA_VERSION_MINOR 0)
set(RAJA_VERSION_PATCHLEVEL 0)

# Find required packages
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  message(STATUS "OpenMP Enabled")
else()
  message(WARNING "OpenMP NOT FOUND")
endif()

# Set Opt as the default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to 'Opt' as none was specified.")
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()

# Configure all our variables
set(RAJA_USE_STL OFF)
## Floating point options
set(RAJA_FP "RAJA_USE_DOUBLE")

## Pointers
set(RAJA_PTR "RAJA_USE_BARE_PTR")
#set(RAJA_USE_BARE_PTR ON)
#set(RAJA_USE_RESTRICT_PTR OFF)
#set(RAJA_USE_RESTRICT_ALIGNED_PTR OFF)
#set(RAJA_USE_PTR_CLASS OFF)

## Fault tolerance
set(RAJA_USE_FT OFF)
set(RAJA_REPORT_FT OFF)

## Platform-specific Constants
set(RAJA_RANGE_ALIGN 4)
set(RAJA_RANGE_MIN_LENGTH 32)
set(RAJA_DATA_ALIGN 64)
set(RAJA_COHERENCE_BLOCK_SIZE 64)

#elseif(CMAKE_CXX_COMPILER_ID MATCHES Clang)
#  set(RAJA_PLATFORM RAJA_PLATFORM_X86_AVX)
#  set(RAJA_COMPILER RAJA_COMPILER_CLANG)
#
#  if(CMAKE_BUILD_TYPE MATCHES Opt)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++0x")
#  elseif(CMAKE_BUILD_TYPE MATCHES Debug)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -std=c++0x -fopenmp")
#  endif()
#endif()

# Configure a header file with all the variables we found.
configure_file(${PROJECT_SOURCE_DIR}/include/RAJA/config.hxx.in
  ${PROJECT_BINARY_DIR}/include/RAJA/config.hxx)
include_directories(${PROJECT_BINARY_DIR}/include/RAJA)

include_directories(include)

add_library(RAJA
  src/AlignedRangeIndexSetBuilders.cxx
  src/DepGraphNode.cxx
  src/IndexSet.cxx
  src/ListSegment.cxx
  src/LockFreeIndexSetBuilders.cxx
  src/MemUtils.cxx
  src/RangeSegment.cxx
  )

file(GLOB files "include/RAJA/*.hxx")
install(FILES ${files} DESTINATION "include/RAJA")
install(FILES ${PROJECT_BINARY_DIR}/include/RAJA/config.hxx DESTINATION "include/RAJA")

install(TARGETS RAJA 
  ARCHIVE DESTINATION lib)

set(BUILD_TESTS On CACHE STRING "Build test applications")
if(BUILD_TESTS)
  add_subdirectory(test)
endif()
