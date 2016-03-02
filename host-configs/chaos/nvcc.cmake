set(RAJA_PLATFORM "RAJA_PLATFORM_X86_SSE" CACHE STRING "")
set(RAJA_COMPILER "RAJA_COMPILER_GNU" CACHE STRING "")

#set(CMAKE_SYSTEM_NAME "CUDA" CACHE STRING "")

#set(CMAKE_C_COMPILER "nvcc" CACHE PATH "")
#set(CMAKE_CXX_COMPILER "nvcc" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/apps/gnu/4.9.3/bin/gcc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "/usr/apps/gnu/4.9.3/bin/g++" CACHE PATH "")

#if(CMAKE_BUILD_TYPE MATCHES Release)
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" CACHE STRING "")
#elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" CACHE STRING "")
#elseif(CMAKE_BUILD_TYPE MATCHES Debug)
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -G -O0 --expt-extended-lambda -arch compute_35 -std=c++11 -Xcompiler -fopenmp -ccbin=/usr/apps/gnu/4.9.3/bin/g++ -x cu" CACHE STRING "")
#  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -G -O0 --expt-extended-lambda -std=c++11 -Xcompiler -fopenmp -ccbin=/usr/apps/gnu/4.9.3/bin/g++" CACHE STRING "")
#  set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -g -G -O0 --expt-extended-lambda -std=c++11 -Xcompiler -fopenmp -ccbin=/usr/apps/gnu/4.9.3/bin/g++" CACHE STRING "")
#endif()

if(CMAKE_BUILD_TYPE MATCHES Release)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -mavx -finline-functions -finline-limit=20000 -std=c++0x" CACHE STRING "")
elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -mavx -finline-functions -finline-limit=20000 -std=c++0x" CACHE STRING "")
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -fpermissive -std=c++0x" CACHE STRING "")
endif()

set(RAJA_USE_CUDA On CACHE BOOL "")

