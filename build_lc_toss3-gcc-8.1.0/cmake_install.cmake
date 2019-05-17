# Install script for directory: /g/g16/hornung1/AASD/RAJA-repo/raja-WORK/RAJA

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/g/g16/hornung1/AASD/RAJA-repo/raja-WORK/RAJA/install_lc_toss3-gcc-8.1.0")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/raja/cmake" TYPE FILE FILES "/g/g16/hornung1/AASD/RAJA-repo/raja-WORK/RAJA/build_lc_toss3-gcc-8.1.0/share/raja/cmake/raja-config.cmake")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/g/g16/hornung1/AASD/RAJA-repo/raja-WORK/RAJA/install_lc_toss3-gcc-8.1.0/lib/pkgconfig/RAJA.pc")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/g/g16/hornung1/AASD/RAJA-repo/raja-WORK/RAJA/install_lc_toss3-gcc-8.1.0/lib/pkgconfig" TYPE FILE FILES "/g/g16/hornung1/AASD/RAJA-repo/raja-WORK/RAJA/build_lc_toss3-gcc-8.1.0/RAJA.pc")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/g/g16/hornung1/AASD/RAJA-repo/raja-WORK/RAJA/build_lc_toss3-gcc-8.1.0/lib/libRAJA.a")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/raja/cmake/RAJA.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/raja/cmake/RAJA.cmake"
         "/g/g16/hornung1/AASD/RAJA-repo/raja-WORK/RAJA/build_lc_toss3-gcc-8.1.0/CMakeFiles/Export/share/raja/cmake/RAJA.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/raja/cmake/RAJA-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/raja/cmake/RAJA.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/raja/cmake" TYPE FILE FILES "/g/g16/hornung1/AASD/RAJA-repo/raja-WORK/RAJA/build_lc_toss3-gcc-8.1.0/CMakeFiles/Export/share/raja/cmake/RAJA.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/raja/cmake" TYPE FILE FILES "/g/g16/hornung1/AASD/RAJA-repo/raja-WORK/RAJA/build_lc_toss3-gcc-8.1.0/CMakeFiles/Export/share/raja/cmake/RAJA-release.cmake")
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/g/g16/hornung1/AASD/RAJA-repo/raja-WORK/RAJA/include/" FILES_MATCHING REGEX "/[^/]*\\.hpp$")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/g/g16/hornung1/AASD/RAJA-repo/raja-WORK/RAJA/tpl/cub/" FILES_MATCHING REGEX "/[^/]*\\.cuh$")
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/RAJA" TYPE FILE FILES
    "/g/g16/hornung1/AASD/RAJA-repo/raja-WORK/RAJA/build_lc_toss3-gcc-8.1.0/include/RAJA/config.hpp"
    "/g/g16/hornung1/AASD/RAJA-repo/raja-WORK/RAJA/include/RAJA/module.modulemap"
    "/g/g16/hornung1/AASD/RAJA-repo/raja-WORK/RAJA/include/RAJA/module.private.modulemap"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/g/g16/hornung1/AASD/RAJA-repo/raja-WORK/RAJA/build_lc_toss3-gcc-8.1.0/blt/thirdparty_builtin/cmake_install.cmake")
  include("/g/g16/hornung1/AASD/RAJA-repo/raja-WORK/RAJA/build_lc_toss3-gcc-8.1.0/blt/tests/smoke/cmake_install.cmake")
  include("/g/g16/hornung1/AASD/RAJA-repo/raja-WORK/RAJA/build_lc_toss3-gcc-8.1.0/test/cmake_install.cmake")
  include("/g/g16/hornung1/AASD/RAJA-repo/raja-WORK/RAJA/build_lc_toss3-gcc-8.1.0/examples/cmake_install.cmake")
  include("/g/g16/hornung1/AASD/RAJA-repo/raja-WORK/RAJA/build_lc_toss3-gcc-8.1.0/exercises/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/g/g16/hornung1/AASD/RAJA-repo/raja-WORK/RAJA/build_lc_toss3-gcc-8.1.0/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
