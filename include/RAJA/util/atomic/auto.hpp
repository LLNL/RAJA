/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining automatic atomic operations.
 *
 ******************************************************************************
 */

#ifndef RAJA_util_atomic_auto_HPP
#define RAJA_util_atomic_auto_HPP

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/config.hpp"
#include "RAJA/util/defines.hpp"


/*!
 * Provides priority between atomic policies that should do the "right thing"
 *
 * If we are in a CUDA __device__ function, then it always uses the cuda_atomic
 * policy.
 *
 * Next, if OpenMP is enabled we always use the omp_atomic, which should
 * generally work everywhere.
 *
 * Finally, we fallback on the builtin_sync_atomic, which is non-standard but
 * has broad compiler support.
 */
#ifdef CUDA_ARCH
  #define RAJA_AUTO_ATOMIC RAJA::cuda_atomic{}
#else
#  ifdef RAJA_ENABLE_OPENMP
  #define RAJA_AUTO_ATOMIC RAJA::omp_atomic{}
#  else
  #define RAJA_AUTO_ATOMIC RAJA::builtin_sync_atomic{}
#  endif
#endif


namespace RAJA
{

//! Atomic policy that automatically does "the right thing"
struct auto_atomic{};


RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_INLINE
RAJA_HOST_DEVICE
constexpr
T atomicAdd(RAJA::auto_atomic, T *acc, T value){
  return RAJA::atomicAdd(RAJA_AUTO_ATOMIC, acc, value);
}


RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_INLINE
RAJA_HOST_DEVICE
constexpr
T atomicSub(RAJA::auto_atomic, T *acc, T value){
  return RAJA::atomicSub(RAJA_AUTO_ATOMIC, acc, value);
}



}  // namespace RAJA

#endif
