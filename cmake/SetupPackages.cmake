###############################################################################
# Copyright (c) 2016, Lawrence Livermore National Security, LLC.
#    
# Produced at the Lawrence Livermore National Laboratory
#    
# LLNL-CODE-689114
# 
# All rights reserved.
#  
# This file is part of RAJA.
#
# For additional details, please also read RAJA/LICENSE.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the disclaimer below.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################

if (ENABLE_CUDA)
  if (ENABLE_OPENMP)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler ${OpenMP_CXX_FLAGS}" CACHE LIST "" FORCE)
  endif()

  if (ENABLE_CUB)
    find_package(CUB)
    if (CUB_FOUND)
      blt_register_library(
        NAME cub
        INCLUDES ${CUB_INCLUDE_DIRS})
    else()
      message(WARNING "Using deprecated Thrust backend for CUDA scans.\n
  Please set CUB_DIR for better scan performance.")
      set(ENABLE_CUB Off)
    endif()
  endif()
endif()

if (ENABLE_TBB)
  find_package(TBB)
  if(TBB_FOUND)
    blt_register_library(
      NAME tbb
      INCLUDES ${TBB_INCLUDE_DIRS})
    message(STATUS "TBB Enabled")
  else()
    message(WARNING "TBB NOT FOUND")
    set(ENABLE_TBB Off)
  endif()
endif ()

if (ENABLE_CHAI)
  message(STATUS "CHAI enabled")
  find_package(chai)
  include_directories(${CHAI_INCLUDE_DIRS})
endif()
