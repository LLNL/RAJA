# Don't allow in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
     message(FATAL_ERROR "In-source builds are not supported. Please remove \
     CMakeCache.txt from the 'src' dir and configure an out-of-source build in \
     another directory.")
 endif()
