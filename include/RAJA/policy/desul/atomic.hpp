//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_atomic_desul_HPP
#define RAJA_policy_atomic_desul_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_DESUL_ATOMICS)

#include "RAJA/util/macros.hpp"

#include "RAJA/policy/atomic_builtin.hpp"

#include "desul/atomics.hpp"

// Default desul options for RAJA
using raja_default_desul_order = desul::MemoryOrderRelaxed;
using raja_default_desul_scope = desul::MemoryScopeDevice;


namespace RAJA
{

RAJA_SUPPRESS_HD_WARN
template <typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T
atomicLoad(AtomicPolicy, T volatile *acc) {
  return desul::atomic_load(const_cast<T*>(acc),
                            raja_default_desul_order{},
                            raja_default_desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE
RAJA_INLINE void
atomicStore(AtomicPolicy, T volatile *acc, T value) {
  return desul::atomic_store(const_cast<T*>(acc),
                             value,
                             raja_default_desul_order{},
                             raja_default_desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T
atomicAdd(AtomicPolicy, T volatile *acc, T value) {
  return desul::atomic_fetch_add(const_cast<T*>(acc),
                                 value,
                                 raja_default_desul_order{},
                                 raja_default_desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T
atomicSub(AtomicPolicy, T volatile *acc, T value) {
  return desul::atomic_fetch_sub(const_cast<T*>(acc),
                                 value,
                                 raja_default_desul_order{},
                                 raja_default_desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicMin(AtomicPolicy, T volatile *acc, T value)
{
  return desul::atomic_fetch_min(const_cast<T*>(acc),
                                 value,
                                 raja_default_desul_order{},
                                 raja_default_desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicMax(AtomicPolicy, T volatile *acc, T value)
{
  return desul::atomic_fetch_max(const_cast<T*>(acc),
                                 value,
                                 raja_default_desul_order{},
                                 raja_default_desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicInc(AtomicPolicy, T volatile *acc)
{
  return desul::atomic_fetch_inc(const_cast<T*>(acc),
                                 raja_default_desul_order{},
                                 raja_default_desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicInc(AtomicPolicy, T volatile *acc, T val)
{
  // See:
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicinc
  return desul::atomic_fetch_inc_mod(const_cast<T*>(acc),
                                          val,
                                          raja_default_desul_order{},
                                          raja_default_desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicDec(AtomicPolicy, T volatile *acc)
{
  return desul::atomic_fetch_dec(const_cast<T*>(acc),
                                 raja_default_desul_order{},
                                 raja_default_desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicDec(AtomicPolicy, T volatile *acc, T val)
{
  // See:
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicdec
  return desul::atomic_fetch_dec_mod(const_cast<T*>(acc),
                                          val,
                                          raja_default_desul_order{},
                                          raja_default_desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicAnd(AtomicPolicy, T volatile *acc, T value)
{
  return desul::atomic_fetch_and(const_cast<T*>(acc),
                                 value,
                                 raja_default_desul_order{},
                                 raja_default_desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicOr(AtomicPolicy, T volatile *acc, T value)
{
  return desul::atomic_fetch_or(const_cast<T*>(acc),
                                value,
                                raja_default_desul_order{},
                                raja_default_desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicXor(AtomicPolicy, T volatile *acc, T value)
{
  return desul::atomic_fetch_xor(const_cast<T*>(acc),
                                 value,
                                 raja_default_desul_order{},
                                 raja_default_desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicExchange(AtomicPolicy, T volatile *acc, T value)
{
  return desul::atomic_exchange(const_cast<T*>(acc),
                                value,
                                raja_default_desul_order{},
                                raja_default_desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE
RAJA_INLINE T atomicCAS(AtomicPolicy, T volatile *acc, T compare, T value)
{
  return desul::atomic_compare_exchange(const_cast<T*>(acc),
                                        compare,
                                        value,
                                        raja_default_desul_order{},
                                        raja_default_desul_scope{});
}

}  // namespace RAJA

#endif  // RAJA_ENABLE_DESUL_ATOMICS
#endif // guard
