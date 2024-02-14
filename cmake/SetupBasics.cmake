###############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
# and other RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

 if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, \
   options are: Debug Release RelWithDebInfo" FORCE)
 endif(NOT CMAKE_BUILD_TYPE)
