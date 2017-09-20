/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining OpenMP atomic operations.
 *
 ******************************************************************************
 */

#ifndef RAJA_policy_openmp_atomic_HPP
#define RAJA_policy_openmp_atomic_HPP

#include "RAJA/config.hpp"

// rely on builtin_atomic when OpenMP can't do the job
#include "RAJA/policy/atomic_builtin.hpp"

#if defined(RAJA_ENABLE_OPENMP)

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

#include "RAJA/util/defines.hpp"


namespace RAJA
{
namespace atomic
{

#ifdef RAJA_COMPILER_MSVC


// For MS Visual C, just default to builtin_atomic for everything
using omp_atomic = builtin_atomic;


#else  // not defined RAJA_COMPILER_MSVC


struct omp_atomic {
};


RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T atomicAdd(omp_atomic, T volatile *acc, T value)
{
  T ret;
#pragma omp atomic capture
  {
    ret = *acc;  // capture old for return value
    *acc += value;
  }
  return ret;
}


RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T atomicSub(omp_atomic, T volatile *acc, T value)
{
  T ret;
#pragma omp atomic capture
  {
    ret = *acc;  // capture old for return value
    *acc -= value;
  }
  return ret;
}


RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T atomicMin(omp_atomic, T volatile *acc, T value)
{
  // OpenMP doesn't define atomic trinary operators so use builtin atomics
  return atomicMin(builtin_atomic{}, acc, value);
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T atomicMax(omp_atomic, T volatile *acc, T value)
{
  // OpenMP doesn't define atomic trinary operators so use builtin atomics
  return atomicMax(builtin_atomic{}, acc, value);
}


RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T atomicInc(omp_atomic, T volatile *acc)
{
  T ret;
#pragma omp atomic capture
  {
    ret = *acc;  // capture old for return value
    *acc += 1;
  }
  return ret;
}


RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T atomicInc(omp_atomic, T volatile *acc, T val)
{
  // OpenMP doesn't define atomic trinary operators so use builtin atomics
  return RAJA::atomic::atomicInc(builtin_atomic{}, acc, val);
}


RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T atomicDec(omp_atomic, T volatile *acc)
{
  T ret;
#pragma omp atomic capture
  {
    ret = *acc;  // capture old for return value
    *acc -= 1;
  }
  return ret;
}


RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T atomicDec(omp_atomic, T volatile *acc, T val)
{
  // OpenMP doesn't define atomic trinary operators so use builtin atomics
  return RAJA::atomic::atomicDec(builtin_atomic{}, acc, val);
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T atomicAnd(omp_atomic, T volatile *acc, T value)
{
  T ret;
#pragma omp atomic capture
  {
    ret = *acc;  // capture old for return value
    *acc &= value;
  }
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T atomicOr(omp_atomic, T volatile *acc, T value)
{
  T ret;
#pragma omp atomic capture
  {
    ret = *acc;  // capture old for return value
    *acc |= value;
  }
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T atomicXor(omp_atomic, T volatile *acc, T value)
{
  T ret;
#pragma omp atomic capture
  {
    ret = *acc;  // capture old for return value
    *acc ^= value;
  }
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T atomicExchange(omp_atomic, T volatile *acc, T value)
{
  T ret;
#pragma omp atomic capture
  {
    ret = *acc;  // capture old for return value
    *acc = value;
  }
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T atomicCAS(omp_atomic, T volatile *acc, T compare, T value)
{
  // OpenMP doesn't define atomic trinary operators so use builtin atomics
  return RAJA::atomic::atomicCAS(builtin_atomic{}, acc, compare, value);
}

#endif  // not defined RAJA_COMPILER_MSVC


}  // namespace atomic
}  // namespace RAJA

#endif  // RAJA_ENABLE_OPENMP
#endif  // guard
