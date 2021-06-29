//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_atomic_desul_cuda_HPP
#define RAJA_policy_atomic_desul_cuda_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "RAJA/util/macros.hpp"

#include "RAJA/policy/atomic_builtin.hpp"

#include "desul/atomics.hpp"

// Default desul options for RAJA
using raja_default_desul_order = desul::MemoryOrderRelaxed;
using raja_default_desul_scope = desul::MemoryScopeDevice;

namespace RAJA
{

/*!
 * Cuda atomic policy for using cuda atomics on the device and
 * the provided Policy on the host
 */
template<typename host_policy>
struct cuda_atomic_explicit{};

/*!
 * Default cuda atomic policy uses cuda atomics on the device and non-atomics
 * on the host
 */

using cuda_atomic = cuda_atomic_explicit<loop_atomic>;

RAJA_SUPPRESS_HD_WARN
template <typename T, typename Policy>
RAJA_HOST_DEVICE
RAJA_INLINE T
atomicAdd(Policy, T volatile *acc, T value)
{
  return desul::atomic_fetch_add(const_cast<T*>(acc),
                                 value,
                                 raja_default_desul_order{},
                                 raja_default_desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename Policy>
RAJA_HOST_DEVICE
RAJA_INLINE T
atomicSub(Policy, T volatile *acc, T value)
{
  return desul::atomic_fetch_sub(const_cast<T*>(acc),
                                 value,
                                 raja_default_desul_order{},
                                 raja_default_desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename Policy>
RAJA_HOST_DEVICE
RAJA_INLINE T
atomicMin(Policy, T volatile *acc, T value)
{
  return desul::atomic_fetch_min(const_cast<T*>(acc),
                                 value,
                                 raja_default_desul_order{},
                                 raja_default_desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename Policy>
RAJA_HOST_DEVICE
RAJA_INLINE T
atomicMax(Policy, T volatile *acc, T value)
{
  return desul::atomic_fetch_max(const_cast<T*>(acc),
                                 value,
                                 raja_default_desul_order{},
                                 raja_default_desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename Policy>
RAJA_HOST_DEVICE
RAJA_INLINE T
atomicInc(Policy, T volatile *acc)
{
  return desul::atomic_fetch_inc(const_cast<T*>(acc),
                                 raja_default_desul_order{},
                                 raja_default_desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename Policy>
RAJA_HOST_DEVICE
RAJA_INLINE T
atomicInc(Policy, T volatile *acc, T val)
{
  // See:
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicinc
  return desul::atomic_wrapping_fetch_inc(const_cast<T*>(acc),
                                          val,
                                          raja_default_desul_order{},
                                          raja_default_desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename Policy>
RAJA_HOST_DEVICE
RAJA_INLINE T
atomicDec(Policy, T volatile *acc)
{
  return desul::atomic_fetch_dec(const_cast<T*>(acc),
                                 raja_default_desul_order{},
                                 raja_default_desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename Policy>
RAJA_HOST_DEVICE
RAJA_INLINE T
atomicDec(Policy, T volatile *acc, T val)
{
  // See:
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicdec
  return desul::atomic_wrapping_fetch_dec(const_cast<T*>(acc),
                                          val,
                                          raja_default_desul_order{},
                                          raja_default_desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename Policy>
RAJA_HOST_DEVICE
RAJA_INLINE T
atomicAnd(Policy, T volatile *acc, T value)
{
  return desul::atomic_fetch_and(const_cast<T*>(acc),
                                 value,
                                 raja_default_desul_order{},
                                 raja_default_desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename Policy>
RAJA_HOST_DEVICE
RAJA_INLINE T
atomicOr(Policy, T volatile *acc, T value)
{
  return desul::atomic_fetch_or(const_cast<T*>(acc),
                                value,
                                raja_default_desul_order{},
                                raja_default_desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename Policy>
RAJA_HOST_DEVICE
RAJA_INLINE T
atomicXor(Policy, T volatile *acc, T value)
{
  return desul::atomic_fetch_xor(const_cast<T*>(acc),
                                 value,
                                 raja_default_desul_order{},
                                 raja_default_desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename Policy>
RAJA_HOST_DEVICE
RAJA_INLINE T
atomicExchange(Policy, T volatile *acc, T value)
{
  return desul::atomic_exchange(const_cast<T*>(acc),
                                value,
                                raja_default_desul_order{},
                                raja_default_desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename Policy>
RAJA_HOST_DEVICE
RAJA_INLINE T
atomicCAS(Policy, T volatile *acc, T compare, T value)
{
  return desul::atomic_compare_exchange(const_cast<T*>(acc),
                                        compare,
                                        value,
                                        raja_default_desul_order{},
                                        raja_default_desul_scope{});
}

}  // namespace RAJA

#endif // RAJA_ENABLE_CUDA
#endif // guard
