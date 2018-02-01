/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining OpenMP atomic operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_openmp_atomic_HPP
#define RAJA_policy_openmp_atomic_HPP

#include "RAJA/config.hpp"

// rely on builtin_atomic when OpenMP can't do the job
#include "RAJA/policy/atomic_builtin.hpp"

#if defined(RAJA_ENABLE_OPENMP)

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
