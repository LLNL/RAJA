###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

configure_file(
     using-with-cmake/host-config.cmake.in
     ${PROJECT_BINARY_DIR}/examples/using-with-cmake/host-config.cmake)

 install( FILES
   using-with-cmake/CMakeLists.txt
   using-with-cmake/using-with-cmake.cpp
   ${PROJECT_BINARY_DIR}/examples/using-with-cmake/host-config.cmake
   DESTINATION examples/RAJA/using-with-cmake)
