set(RAJA_PLATFORM RAJA_PLATFORM_X86_AVX)
set(RAJA_COMPILER RAJA_COMPILER_GNU)

if(CMAKE_BUILD_TYPE MATCHES Opt)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -mavx -finline-functions -finline-limit=20000 -std=c++0x")
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -fpermissive -std=c++0x")
endif()
