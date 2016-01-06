set(RAJA_PLATFORM "RAJA_PLATFORM_X86_AVX" CACHE STRING "")
set(RAJA_COMPILER "RAJA_COMPILER_ICC" CACHE STRING "")

set(CMAKE_C_COMPILER "/usr/local/bin/icc-16.0.109" CACHE PATH "")

# cpp compiler used by spack
set(CMAKE_CXX_COMPILER "/usr/local/bin/icpc-16.0.109" CACHE PATH "")

message(${CMAKE_BUILD_TYPE})

if(CMAKE_BUILD_TYPE MATCHES Release)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx -inline-max-total-size=20000 -inline-forceinline -ansi-alias -std=c++0x" CACHE STRING "")
elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx -inline-max-total-size=20000 -inline-forceinline -ansi-alias -std=c++0x" CACHE STRING "")
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -std=c++0x" CACHE STRING "")
endif()
