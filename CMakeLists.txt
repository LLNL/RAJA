cmake_minimum_required (VERSION 3.1)

project(RAJA LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules" ${CMAKE_MODULE_PATH})

enable_testing()

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

if (RAJA_USE_CUDA)
  find_package(CUDA)
  if(CUDA_FOUND)
    message(STATUS "CUDA")
    #set(CUDA_NVCC_FLAGS -O2; --expt-extended-lambda; -restrict; -arch compute_35; -std=c++11; -Xcompiler; -x cu)
    set(CUDA_NVCC_FLAGS "-arch compute_35 -std=c++11 --expt-extended-lambda -x=cu")
    #set (CUDA_PROPAGATE_HOST_FLAGS OFF)
    include_directories(${CUDA_INCLUDE_DIRS})
  endif()
endif()

# Used for timing
find_library(RT_LIBRARIES rt)


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
if (RAJA_USE_CUDA)
  set(RAJA_PTR "RAJA_USE_BARE_PTR")
else ()
  set(RAJA_PTR "RAJA_USE_RESTRICT_PTR")
endif()
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

# Timer
add_definitions(-DRAJA_USE_GETTIME)

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
