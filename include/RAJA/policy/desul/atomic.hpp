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

#include "RAJA/policy/atomic_builtin.hpp"
#include "RAJA/policy/desul/policy.hpp"
#include "RAJA/util/macros.hpp"
#include "desul/atomics.hpp"

// Default desul options for RAJA
using raja_default_desul_order = desul::MemoryOrderRelaxed;
using raja_default_desul_scope = desul::MemoryScopeDevice;


namespace RAJA
{

namespace detail
{
template <typename T>
struct DesulAtomicPolicy {
  using memory_order = raja_default_desul_order;
  using memory_scope = raja_default_desul_scope;
};

template <typename OrderingPolicy, typename ScopePolicy>
struct DesulAtomicPolicy<detail_atomic_t<OrderingPolicy, ScopePolicy>> {
  using memory_order = OrderingPolicy;
  using memory_scope = ScopePolicy;
};

}  // namespace detail

RAJA_SUPPRESS_HD_WARN
template <typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicAdd(AtomicPolicy, T volatile *acc, T value)
{
  using desul_order =
      typename detail::DesulAtomicPolicy<AtomicPolicy>::memory_order;
  using desul_scope =
      typename detail::DesulAtomicPolicy<AtomicPolicy>::memory_scope;
  return desul::atomic_fetch_add(const_cast<T *>(acc),
                                 value,
                                 desul_order{},
                                 desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicSub(AtomicPolicy, T volatile *acc, T value)
{
  using desul_order =
      typename detail::DesulAtomicPolicy<AtomicPolicy>::memory_order;
  using desul_scope =
      typename detail::DesulAtomicPolicy<AtomicPolicy>::memory_scope;
  return desul::atomic_fetch_sub(const_cast<T *>(acc),
                                 value,
                                 desul_order{},
                                 desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicMin(AtomicPolicy, T volatile *acc, T value)
{
  using desul_order =
      typename detail::DesulAtomicPolicy<AtomicPolicy>::memory_order;
  using desul_scope =
      typename detail::DesulAtomicPolicy<AtomicPolicy>::memory_scope;
  return desul::atomic_fetch_min(const_cast<T *>(acc),
                                 value,
                                 desul_order{},
                                 desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicMax(AtomicPolicy, T volatile *acc, T value)
{
  using desul_order =
      typename detail::DesulAtomicPolicy<AtomicPolicy>::memory_order;
  using desul_scope =
      typename detail::DesulAtomicPolicy<AtomicPolicy>::memory_scope;
  return desul::atomic_fetch_max(const_cast<T *>(acc),
                                 value,
                                 desul_order{},
                                 desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicInc(AtomicPolicy, T volatile *acc)
{
  using desul_order =
      typename detail::DesulAtomicPolicy<AtomicPolicy>::memory_order;
  using desul_scope =
      typename detail::DesulAtomicPolicy<AtomicPolicy>::memory_scope;
  return desul::atomic_fetch_inc(const_cast<T *>(acc),
                                 desul_order{},
                                 desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicInc(AtomicPolicy, T volatile *acc, T val)
{
  using desul_order =
      typename detail::DesulAtomicPolicy<AtomicPolicy>::memory_order;
  using desul_scope =
      typename detail::DesulAtomicPolicy<AtomicPolicy>::memory_scope;
  // See:
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicinc
  return desul::atomic_fetch_inc_mod(const_cast<T *>(acc),
                                     val,
                                     desul_order{},
                                     desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicDec(AtomicPolicy, T volatile *acc)
{
  using desul_order =
      typename detail::DesulAtomicPolicy<AtomicPolicy>::memory_order;
  using desul_scope =
      typename detail::DesulAtomicPolicy<AtomicPolicy>::memory_scope;
  return desul::atomic_fetch_dec(const_cast<T *>(acc),
                                 desul_order{},
                                 desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicDec(AtomicPolicy, T volatile *acc, T val)
{
  using desul_order =
      typename detail::DesulAtomicPolicy<AtomicPolicy>::memory_order;
  using desul_scope =
      typename detail::DesulAtomicPolicy<AtomicPolicy>::memory_scope;
  // See:
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicdec
  return desul::atomic_fetch_dec_mod(const_cast<T *>(acc),
                                     val,
                                     desul_order{},
                                     desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicAnd(AtomicPolicy, T volatile *acc, T value)
{
  using desul_order =
      typename detail::DesulAtomicPolicy<AtomicPolicy>::memory_order;
  using desul_scope =
      typename detail::DesulAtomicPolicy<AtomicPolicy>::memory_scope;
  return desul::atomic_fetch_and(const_cast<T *>(acc),
                                 value,
                                 desul_order{},
                                 desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicOr(AtomicPolicy, T volatile *acc, T value)
{
  using desul_order =
      typename detail::DesulAtomicPolicy<AtomicPolicy>::memory_order;
  using desul_scope =
      typename detail::DesulAtomicPolicy<AtomicPolicy>::memory_scope;
  return desul::atomic_fetch_or(const_cast<T *>(acc),
                                value,
                                desul_order{},
                                desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicXor(AtomicPolicy, T volatile *acc, T value)
{
  using desul_order =
      typename detail::DesulAtomicPolicy<AtomicPolicy>::memory_order;
  using desul_scope =
      typename detail::DesulAtomicPolicy<AtomicPolicy>::memory_scope;
  return desul::atomic_fetch_xor(const_cast<T *>(acc),
                                 value,
                                 desul_order{},
                                 desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE RAJA_INLINE T atomicExchange(AtomicPolicy,
                                              T volatile *acc,
                                              T value)
{
  using desul_order =
      typename detail::DesulAtomicPolicy<AtomicPolicy>::memory_order;
  using desul_scope =
      typename detail::DesulAtomicPolicy<AtomicPolicy>::memory_scope;
  return desul::atomic_exchange(const_cast<T *>(acc),
                                value,
                                desul_order{},
                                desul_scope{});
}

RAJA_SUPPRESS_HD_WARN
template <typename AtomicPolicy, typename T>
RAJA_HOST_DEVICE RAJA_INLINE T
atomicCAS(AtomicPolicy, T volatile *acc, T compare, T value)
{
  using desul_order =
      typename detail::DesulAtomicPolicy<AtomicPolicy>::memory_order;
  using desul_scope =
      typename detail::DesulAtomicPolicy<AtomicPolicy>::memory_scope;
  return desul::atomic_compare_exchange(
      const_cast<T *>(acc), compare, value, desul_order{}, desul_scope{});
}

}  // namespace RAJA

#endif  // RAJA_ENABLE_DESUL_ATOMICS
#endif  // guard
