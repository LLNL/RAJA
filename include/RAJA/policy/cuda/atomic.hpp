/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining atomic operations for CUDA
 *
 ******************************************************************************
 */

#ifndef RAJA_policy_cuda_atomic_HPP
#define RAJA_policy_cuda_atomic_HPP

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

#include <stdexcept>

#if defined(RAJA_ENABLE_CUDA)


#if __CUDA_ARCH__ < 600
#define RAJA_CUDA_DOUBLE_ATOMIC_CAS
#endif

//
// Note: I would much rather all of these functions be device-only, but ran
//       into issues with their call sites... so instead we have this dumb
//       #ifdef __CUDA_ARCH__ stuff for essentially device-only code. (AJK)
//



namespace RAJA
{
struct cuda_atomic{};



/*!
 * Catch-all policy passes off to CUDA's builtin atomics.
 *
 * This catch-all will only work for types supported by the compiler.
 * Specialization below can adapt for some unsupported types.
 */
RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_INLINE
__host__ __device__
T atomicAdd(cuda_atomic, T *acc, T value){
#ifdef __CUDA_ARCH__

  return ::atomicAdd(acc, value);
#else
  throw std::logic_error("Cannot call cuda_atomic operations on host");
#endif
}



// Before Pascal's, no native support for double-precision atomic add
// So we use the CAS approach
#ifdef RAJA_CUDA_DOUBLE_ATOMIC_CAS

template<>
RAJA_SUPPRESS_HD_WARN
RAJA_INLINE
__host__ __device__
double atomicAdd<double>(cuda_atomic, double *acc, double value){
#ifdef __CUDA_ARCH__
  unsigned long long oldval, newval, readback;
  oldval = __double_as_longlong(*acc);
  newval = __double_as_longlong(__longlong_as_double(oldval) + value);
  while ((readback = ::atomicCAS((unsigned long long *)acc, oldval, newval))
         != oldval) {
    oldval = readback;
    newval = __double_as_longlong(__longlong_as_double(oldval) + value);
  }
  return __longlong_as_double(oldval);
#else
  throw std::logic_error("Cannot call cuda_atomic operations on host");
#endif
}

#endif







RAJA_SUPPRESS_HD_WARN
template<typename T>
RAJA_INLINE
__host__ __device__
constexpr
T atomicSub(cuda_atomic, T *acc, T value){
  return RAJA::atomicAdd(cuda_atomic{}, acc, -value);
}







}  // namespace RAJA

#endif // RAJA_ENABLE_CUDA
#endif // guard
