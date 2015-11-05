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
set(RAJA_USE_DOUBLE ON)
set(RAJA_USE_FLOAT OFF)
## Pointers
set(RAJA_USE_BARE_PTR ON)
set(RAJA_USE_RESTRICT_PTR OFF)
set(RAJA_USE_RESTRICT_ALIGNED_PTR OFF)
set(RAJA_USE_PTR_CLASS OFF)
## Fault tolerance
set(RAJA_USE_FT OFF)
set(RAJA_REPORT_FT OFF)
## GPUs
set(RAJA_USE_CUDA OFF)

# Add compiler specific flags
if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
  set(RAJA_PLATFORM RAJA_PLATFORM_X86_AVX)
  set(RAJA_COMPILER RAJA_COMPILER_GNU)

  if(CMAKE_BUILD_TYPE MATCHES Opt)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -mavx -finline-functions -finline-limit=20000 -std=c++0x")
  elseif(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -fpermissive -std=c++0x")
  endif()

elseif(CMAKE_CXX_COMPILER_ID MATCHES Intel)
  set(RAJA_PLATFORM RAJA_PLATFORM_X86_AVX)
  set(RAJA_COMPILER RAJA_COMPILER_ICC)

  if(CMAKE_BUILD_TYPE MATCHES Opt)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx -inline-max-total-size=20000 -inline-forceinline -ansi-alias -std=c++0x")
  elseif(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -std=c++0x")
  endif()

elseif(CMAKE_CXX_COMPILER_ID MATCHES Clang)
  set(RAJA_PLATFORM RAJA_PLATFORM_X86_AVX)
  set(RAJA_COMPILER RAJA_COMPILER_CLANG)

  if(CMAKE_BUILD_TYPE MATCHES Opt)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++0x")
  elseif(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -std=c++0x -fopenmp")
  endif()

endif()


# Configure a header file with all the variables we found.
configure_file(${PROJECT_SOURCE_DIR}/includes/RAJA/config.hxx.in
  ${PROJECT_BINARY_DIR}/include/RAJA/config.hxx)
include_directories(${PROJECT_BINARY_DIR}/include/RAJA)

include_directories(includes)

add_library(RAJA
  sources/AlignedRangeIndexSetBuilders.cxx
  sources/DepGraphNode.cxx
  sources/IndexSet.cxx
  sources/ListSegment.cxx
  sources/LockFreeIndexSetBuilders.cxx
  sources/MemUtils.cxx
  sources/RangeSegment.cxx
)

file(GLOB files "includes/RAJA/*.hxx")
install(FILES ${files} DESTINATION "include/RAJA")
install(FILES ${PROJECT_BINARY_DIR}/include/RAJA/config.hxx DESTINATION "include/RAJA")

install(TARGETS RAJA 
  ARCHIVE DESTINATION lib)

set(BUILD_TESTS On CACHE STRING "Build test applications")
if(BUILD_TESTS)
  add_subdirectory(test)
endif()
