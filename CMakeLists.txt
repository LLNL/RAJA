cmake_minimum_required (VERSION 3.0)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules" ${CMAKE_MODULE_PATH})

project(RAJA LANGUAGES CXX)

enable_testing()

# Set version number
set(RAJA_VERSION_MAJOR 1)
set(RAJA_VERSION_MINOR 0)
set(RAJA_VERSION_PATCHLEVEL 0)

# We need to find the packages _after_ the CMake starts up, so this isn't in
# a host-config file.
if (RAJA_USE_OPENMP)
  find_package(OpenMP REQUIRED)
  if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    list(APPEND RAJA_NVCC_FLAGS -Xcompiler ${OpenMP_CXX_FLAGS})
    message(STATUS "OpenMP Enabled")
  else()
    message(WARNING "OpenMP NOT FOUND")
  endif()
endif()

if (RAJA_USE_CUDA)
  find_package(CUDA)
  if(CUDA_FOUND)
    message(STATUS "CUDA")
    set (CUDA_NVCC_FLAGS ${RAJA_NVCC_FLAGS})
    set (CUDA_PROPAGATE_HOST_FLAGS OFF)
    include_directories(${CUDA_INCLUDE_DIRS})
  endif()
endif()

# Used for timing
find_library(RT_LIBRARIES rt)


# Set Release as the default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

#
# Configure all variables in the code 
#

## Floating point options
set(RAJA_FP "RAJA_USE_DOUBLE")
#set(RAJA_FP "RAJA_USE_FLOAT")

## Pointer options
if (RAJA_USE_CUDA)
  set(RAJA_PTR "RAJA_USE_BARE_PTR")
else ()
  set(RAJA_PTR "RAJA_USE_RESTRICT_PTR")
endif()
#set(RAJA_USE_BARE_PTR ON)
#set(RAJA_USE_RESTRICT_PTR OFF)
#set(RAJA_USE_RESTRICT_ALIGNED_PTR OFF)
#set(RAJA_USE_PTR_CLASS OFF)

## Fault tolerance options
set(RAJA_USE_FT OFF)
set(RAJA_REPORT_FT OFF)

## Timer options
add_definitions(-DRAJA_USE_GETTIME)
#add_definitions(-DRAJA_USE_CLOCK)
#add_definitions(-DRAJA_USE_CYCLE)

# Configure a header file with all the variables we found.
configure_file(${PROJECT_SOURCE_DIR}/include/RAJA/config.hxx.in
  ${PROJECT_BINARY_DIR}/include/RAJA/config.hxx)
include_directories(${PROJECT_BINARY_DIR}/include/RAJA)
include_directories(${PROJECT_BINARY_DIR}/include)

include_directories(include)

install(DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN *.hxx)
install(FILES ${PROJECT_BINARY_DIR}/include/RAJA/config.hxx DESTINATION "include/RAJA")


if (RAJA_USE_CUDA)
cuda_add_library(RAJA
    src/AlignedRangeIndexSetBuilders.cxx
    src/DepGraphNode.cxx
    src/IndexSet.cxx
    src/ListSegment.cxx
    src/LockFreeIndexSetBuilders.cxx
    src/MemUtils_CPU.cxx
    src/MemUtils_CUDA.cxx
    src/RangeSegment.cxx
    src/ThreadUtils_CPU.cxx
  )
else()
add_library(RAJA
    src/AlignedRangeIndexSetBuilders.cxx
    src/DepGraphNode.cxx
    src/IndexSet.cxx
    src/ListSegment.cxx
    src/LockFreeIndexSetBuilders.cxx
    src/MemUtils_CPU.cxx
    src/MemUtils_CUDA.cxx
    src/RangeSegment.cxx
    src/ThreadUtils_CPU.cxx
  )
endif()

install(TARGETS RAJA ARCHIVE DESTINATION lib)

set(BUILD_TESTS On CACHE STRING "Build test applications")
if(BUILD_TESTS)
  add_subdirectory(test)
endif()
