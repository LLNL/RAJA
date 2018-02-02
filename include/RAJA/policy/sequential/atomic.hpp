/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining sequential atomic operations.
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

#ifndef RAJA_policy_sequential_atomic_HPP
#define RAJA_policy_sequential_atomic_HPP

#include "RAJA/config.hpp"
#include "RAJA/util/defines.hpp"

namespace RAJA
{
namespace atomic
{

struct seq_atomic {
};


RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T atomicAdd(seq_atomic, T volatile *acc, T value)
{
  T ret = *acc;
  *acc += value;
  return ret;
}


RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T atomicSub(seq_atomic, T volatile *acc, T value)
{
  T ret = *acc;
  *acc -= value;
  return ret;
}


RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T atomicMin(seq_atomic, T volatile *acc, T value)
{
  T ret = *acc;
  *acc = ret < value ? ret : value;
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T atomicMax(seq_atomic, T volatile *acc, T value)
{
  T ret = *acc;
  *acc = ret > value ? ret : value;
  return ret;
}


RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T atomicInc(seq_atomic, T volatile *acc)
{
  T ret = *acc;
  (*acc) += 1;
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T atomicInc(seq_atomic, T volatile *acc, T val)
{
  T old = *acc;
  (*acc) = ((old >= val) ? 0 : (old + 1));
  return old;
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T atomicDec(seq_atomic, T volatile *acc)
{
  T ret = *acc;
  (*acc) -= 1;
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T atomicDec(seq_atomic, T volatile *acc, T val)
{
  T old = *acc;
  (*acc) = (((old == 0) | (old > val)) ? val : (old - 1));
  return old;
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T atomicAnd(seq_atomic, T volatile *acc, T value)
{
  T ret = *acc;
  *acc &= value;
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T atomicOr(seq_atomic, T volatile *acc, T value)
{
  T ret = *acc;
  *acc |= value;
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T atomicXor(seq_atomic, T volatile *acc, T value)
{
  T ret = *acc;
  *acc ^= value;
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T atomicExchange(seq_atomic, T volatile *acc, T value)
{
  T ret = *acc;
  *acc = value;
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T atomicCAS(seq_atomic, T volatile *acc, T compare, T value)
{
  T ret = *acc;
  *acc = ret == compare ? value : ret;
  return ret;
}


}  // namespace atomic
}  // namespace RAJA


#endif  // guard
