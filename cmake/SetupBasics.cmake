# Don't allow in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
     message(FATAL_ERROR "In-source builds are not supported. Please remove \
     CMakeCache.txt from the 'src' dir and configure an out-of-source build in \
     another directory.")
 endif()

 if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, \
   options are: Debug Release RelWithDebInfo" FORCE)
 endif(NOT CMAKE_BUILD_TYPE)
