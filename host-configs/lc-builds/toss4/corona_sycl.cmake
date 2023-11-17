###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set(RAJA_COMPILER "RAJA_COMPILER_CLANG" CACHE STRING "")

set(CMAKE_CXX_COMPILER "clang++" CACHE PATH "")
#set(CMAKE_CXX_COMPILER "dpcpp" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -L${SYCL_LIB_PATH} -fsycl -fsycl-unnamed-lambda -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx90a" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -L${SYCL_LIB_PATH} -fsycl -fsycl-unnamed-lambda -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx90a" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -L${SYCL_LIB_PATH} -fsycl -fsycl-unnamed-lambda -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx90a" CACHE STRING "")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -std=c++17 -fsycl -fsycl-unnamed-lambda --gcc-toolchain=/usr/tce/packages/gcc/gcc-7.1.0" CACHE STRING "")
#set(CMAKE_CXX_FLAGS_RELWITHDEBINFO " -O3 -g  -std=c++17 -fsycl -fsycl-unnamed-lambda --gcc-toolchain=/usr/tce/packages/gcc/gcc-7.1.0" CACHE STRING "")
#set(CMAKE_CXX_FLAGS_DEBUG " -O0 -g -std=c++17 -fsycl -fsycl-unnamed-lambda --gcc-toolchain=/usr/tce/packages/gcc/gcc-7.1.0" CACHE STRING "")
#set(CMAKE_CXX_LINK_FLAGS "-fsycl -Wl,-rpath,/usr/tce/packages/oneapi/oneapi-2021.2/compiler/2021.2.0/linux/compiler/lib/intel64_lin/"  CACHE STRING "")

set(RAJA_HOST_CONFIG_LOADED On CACHE BOOL "")
