set(RAJA_PLATFORM "RAJA_PLATFORM_X86_AVX" CACHE STRING "")
set(RAJA_COMPILER "RAJA_COMPILER_GNU" CACHE STRING "")

set(CMAKE_C_COMPILER "/usr/apps/gnu/4.9.3/bin/gcc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "/usr/apps/gnu/4.9.3/bin/g++" CACHE PATH "")

message(${CMAKE_BUILD_TYPE})

if(CMAKE_BUILD_TYPE MATCHES Release)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -mavx -finline-functions -finline-limit=20000 -std=c++0x" CACHE STRING "")
elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -mavx -finline-functions -finline-limit=20000 -std=c++0x" CACHE STRING "")
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -fpermissive -std=c++0x" CACHE STRING "")
endif()

set(RAJA_USE_OPENMP On CACHE BOOL "")
