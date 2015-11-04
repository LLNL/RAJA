cmake_minimum_required (VERSION 3.1)

project(RAJA VERSION 0.0 LANGUAGES CXX)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  message(STATUS "OpenMP Enabled")
else()
  message(WARNING "OpenMP NOT FOUND")
endif()

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to 'Opt' as none was specified.")
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
  add_definitions(-DRAJA_PLATFORM_X86_AVX -DRAJA_COMPILER_GNU)
  if(CMAKE_BUILD_TYPE MATCHES Opt)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -mavx -finline-functions -finline-limit=20000 -std=c++0x")
  elseif(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -fpermissive -std=c++0x")
  endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES Intel)
  add_definitions(-DRAJA_PLATFORM_X86_AVX -DRAJA_COMPILER_ICC)
  if(CMAKE_BUILD_TYPE MATCHES Opt)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx -inline-max-total-size=20000 -inline-forceinline -ansi-alias -std=c++0x")
  elseif(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -std=c++0x")
  endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES Clang)
  add_definitions(-DRAJA_PLATFORM_X86_AVX -DRAJA_COMPILER_CLANG)
  if(CMAKE_BUILD_TYPE MATCHES Opt)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++0x")
  elseif(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -std=c++0x -fopenmp")
  endif()
endif()

add_definitions(-DRAJA_USE_DOUBLE -DRAJA_USE_RESTRICT_PTR)

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

install(TARGETS RAJA 
  ARCHIVE DESTINATION lib)
