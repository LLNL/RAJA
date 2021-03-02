/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining atomic operations for HIP
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_hip_atomic_HPP
#define RAJA_policy_hip_atomic_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include <stdexcept>
#include <type_traits>
#include "hip/hip_runtime.h"

#include "RAJA/policy/loop/atomic.hpp"
#include "RAJA/policy/sequential/atomic.hpp"
#include "RAJA/policy/atomic_builtin.hpp"
#if defined(RAJA_ENABLE_OPENMP)
#include "RAJA/policy/openmp/atomic.hpp"
#endif

#include "RAJA/util/Operators.hpp"
#include "RAJA/util/TypeConvert.hpp"
#include "RAJA/util/macros.hpp"


namespace RAJA
{

namespace detail
{

/*!
 * Generic impementation of atomic 32-bit or 64-bit compare and swap primitive.
 * Implementation uses the existing HIP supplied unsigned 32-bit and 64-bit
 * CAS operators.
 * Returns the value that was stored before this operation.
 */
RAJA_INLINE __device__ unsigned hip_atomic_CAS(
    unsigned volatile *acc,
    unsigned compare,
    unsigned value)
{
  return ::atomicCAS((unsigned *)acc, compare, value);
}
///
RAJA_INLINE __device__ unsigned long long hip_atomic_CAS(
    unsigned long long volatile *acc,
    unsigned long long compare,
    unsigned long long value)
{
  return ::atomicCAS((unsigned long long *)acc, compare, value);
}
///
template <typename T>
RAJA_INLINE __device__
typename std::enable_if<sizeof(T) == sizeof(unsigned), T>::type
hip_atomic_CAS(T volatile *acc, T compare, T value)
{
  return RAJA::util::reinterp_A_as_B<unsigned, T>(
      hip_atomic_CAS((unsigned volatile *)acc,
          RAJA::util::reinterp_A_as_B<T, unsigned>(compare),
          RAJA::util::reinterp_A_as_B<T, unsigned>(value)));
}
///
template <typename T>
RAJA_INLINE __device__
typename std::enable_if<sizeof(T) == sizeof(unsigned long long), T>::type
hip_atomic_CAS(T volatile *acc, T compare, T value)
{
  return RAJA::util::reinterp_A_as_B<unsigned long long, T>(
      hip_atomic_CAS((unsigned long long volatile *)acc,
          RAJA::util::reinterp_A_as_B<T, unsigned long long>(compare),
          RAJA::util::reinterp_A_as_B<T, unsigned long long>(value)));
}

template <size_t BYTES>
struct HipAtomicCAS {
};


template <>
struct HipAtomicCAS<4> {

  /*!
   * Generic impementation of any atomic 32-bit operator.
   * Implementation uses the existing HIP supplied unsigned 32-bit CAS
   * operator. Returns the OLD value that was replaced by the result of this
   * operation.
   */
  template <typename T, typename OPER>
  RAJA_INLINE __device__ T operator()(T volatile *acc, OPER const &oper) const
  {
    // asserts in RAJA::util::reinterp_T_as_u and RAJA::util::reinterp_u_as_T
    // will enforce 32-bit T
    unsigned oldval, newval, readback;
    oldval = RAJA::util::reinterp_A_as_B<T, unsigned>(*acc);
    newval = RAJA::util::reinterp_A_as_B<T, unsigned>(
        oper(RAJA::util::reinterp_A_as_B<unsigned, T>(oldval)));
    while ((readback = hip_atomic_CAS((unsigned volatile*)acc, oldval, newval)) !=
           oldval) {
      oldval = readback;
      newval = RAJA::util::reinterp_A_as_B<T, unsigned>(
          oper(RAJA::util::reinterp_A_as_B<unsigned, T>(oldval)));
    }
    return RAJA::util::reinterp_A_as_B<unsigned, T>(oldval);
  }
};

template <>
struct HipAtomicCAS<8> {

  /*!
   * Generic impementation of any atomic 64-bit operator.
   * Implementation uses the existing HIP supplied unsigned 64-bit CAS
   * operator. Returns the OLD value that was replaced by the result of this
   * operation.
   */
  template <typename T, typename OPER>
  RAJA_INLINE __device__ T operator()(T volatile *acc, OPER const &oper) const
  {
    // asserts in RAJA::util::reinterp_T_as_u and RAJA::util::reinterp_u_as_T
    // will enforce 64-bit T
    unsigned long long oldval, newval, readback;
    oldval = RAJA::util::reinterp_A_as_B<T, unsigned long long>(*acc);
    newval = RAJA::util::reinterp_A_as_B<T, unsigned long long>(
        oper(RAJA::util::reinterp_A_as_B<unsigned long long, T>(oldval)));
    while (
        (readback = hip_atomic_CAS((unsigned long long volatile*)acc, oldval, newval)) !=
        oldval) {
      oldval = readback;
      newval = RAJA::util::reinterp_A_as_B<T, unsigned long long>(
          oper(RAJA::util::reinterp_A_as_B<unsigned long long, T>(oldval)));
    }
    return RAJA::util::reinterp_A_as_B<unsigned long long, T>(oldval);
  }
};


/*!
 * Generic impementation of any atomic 32-bit or 64-bit operator that can be
 * implemented using a compare and swap primitive.
 * Implementation uses the existing HIP supplied unsigned 32-bit and 64-bit
 * CAS operators.
 * Returns the OLD value that was replaced by the result of this operation.
 */
template <typename T, typename OPER>
RAJA_INLINE __device__ T hip_atomic_CAS_oper(T volatile *acc, OPER &&oper)
{
  HipAtomicCAS<sizeof(T)> cas;
  return cas(acc, std::forward<OPER>(oper));
}


/*!
 * Catch-all policy passes off to HIP's builtin atomics.
 *
 * This catch-all will only work for types supported by the compiler.
 * Specialization below can adapt for some unsupported types.
 *
 * These are atomic in hip device code and non-atomic otherwise
 */
template <typename T>
RAJA_INLINE __device__ T hip_atomicAdd(T volatile *acc, T value)
{
  return hip_atomic_CAS_oper(acc, [=] __device__(T a) {
    return a + value;
  });
}

#if __HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__
// 32-bit signed atomicAdd support by HIP
template <>
RAJA_INLINE __device__ int hip_atomicAdd<int>(int volatile *acc,
                                          int value)
{
  return ::atomicAdd((int *)acc, value);
}


// 32-bit unsigned atomicAdd support by HIP
template <>
RAJA_INLINE __device__ unsigned hip_atomicAdd<unsigned>(unsigned volatile *acc,
                                                    unsigned value)
{
  return ::atomicAdd((unsigned *)acc, value);
}
#endif

// 64-bit unsigned atomicAdd support by HIP
#if __HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__
template <>
RAJA_INLINE __device__ unsigned long long hip_atomicAdd<unsigned long long>(
    unsigned long long volatile *acc,
    unsigned long long value)
{
  return ::atomicAdd((unsigned long long *)acc, value);
}
#endif


// 32-bit float atomicAdd support by HIP
#if __HIP_ARCH_HAS_FLOAT_ATOMIC_ADD__
template <>
RAJA_INLINE __device__ float hip_atomicAdd<float>(float volatile *acc,
                                              float value)
{
  return ::atomicAdd((float *)acc, value);
}
#endif

template <typename T>
RAJA_INLINE __device__ T hip_atomicSub(T volatile *acc, T value)
{
  return hip_atomic_CAS_oper(acc, [=] __device__(T a) {
    return a - value;
  });
}

#if __HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__
// 32-bit signed atomicSub support by HIP
template <>
RAJA_INLINE __device__ int hip_atomicSub<int>(int volatile *acc,
                                          int value)
{
  return ::atomicSub((int *)acc, value);
}

// 32-bit unsigned atomicSub support by HIP
template <>
RAJA_INLINE __device__ unsigned hip_atomicSub<unsigned>(unsigned volatile *acc,
                                                    unsigned value)
{
  return ::atomicSub((unsigned *)acc, value);
}
#endif

template <typename T>
RAJA_INLINE __device__ T hip_atomicMin(T volatile *acc, T value)
{
  return hip_atomic_CAS_oper(acc, [=] __device__(T a) {
    return value < a ? value : a;
  });
}

#if __HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__
// 32-bit signed atomicMin support by HIP
template <>
RAJA_INLINE __device__ int hip_atomicMin<int>(int volatile *acc,
                                          int value)
{
  return ::atomicMin((int *)acc, value);
}


// 32-bit unsigned atomicMin support by HIP
template <>
RAJA_INLINE __device__ unsigned hip_atomicMin<unsigned>(unsigned volatile *acc,
                                                    unsigned value)
{
  return ::atomicMin((unsigned *)acc, value);
}
#endif

#if __HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__
template <>
RAJA_INLINE __device__ unsigned long long hip_atomicMin<unsigned long long>(
    unsigned long long volatile *acc,
    unsigned long long value)
{
  return ::atomicMin((unsigned long long *)acc, value);
}
#endif

template <typename T>
RAJA_INLINE __device__ T hip_atomicMax(T volatile *acc, T value)
{
  return hip_atomic_CAS_oper(acc, [=] __device__(T a) {
    return value > a ? value : a;
  });
}

#if __HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__
// 32-bit signed atomicMax support by HIP
template <>
RAJA_INLINE __device__ int hip_atomicMax<int>(int volatile *acc,
                                          int value)
{
  return ::atomicMax((int *)acc, value);
}


// 32-bit unsigned atomicMax support by HIP
template <>
RAJA_INLINE __device__ unsigned hip_atomicMax<unsigned>(unsigned volatile *acc,
                                                    unsigned value)
{
  return ::atomicMax((unsigned *)acc, value);
}
#endif

#if __HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__
template <>
RAJA_INLINE __device__ unsigned long long hip_atomicMax<unsigned long long>(
    unsigned long long volatile *acc,
    unsigned long long value)
{
  return ::atomicMax((unsigned long long *)acc, value);
}
#endif

template <typename T>
RAJA_INLINE __device__ T hip_atomicInc(T volatile *acc, T val)
{
  return hip_atomic_CAS_oper(acc, [=] __device__(T old) {
    return ((old >= val) ? 0 : (old + 1));
  });
}

#if __HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__
// 32-bit unsigned atomicInc support by HIP
template <>
RAJA_INLINE __device__ unsigned hip_atomicInc<unsigned>(unsigned volatile *acc,
                                                    unsigned value)
{
  return ::atomicInc((unsigned *)acc, value);
}
#endif

template <typename T>
RAJA_INLINE __device__ T hip_atomicInc(T volatile *acc)
{
  return hip_atomic_CAS_oper(acc,
                                      [=] __device__(T a) { return a + 1; });
}


template <typename T>
RAJA_INLINE __device__ T hip_atomicDec(T volatile *acc, T val)
{
  // See:
  // http://docs.nvidia.com/hip/hip-c-programming-guide/index.html#atomicdec
  return hip_atomic_CAS_oper(acc, [=] __device__(T old) {
    return (((old == 0) | (old > val)) ? val : (old - 1));
  });
}

#if __HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__
// 32-bit unsigned atomicDec support by HIP
template <>
RAJA_INLINE __device__ unsigned hip_atomicDec<unsigned>(unsigned volatile *acc,
                                                    unsigned value)
{
  return ::atomicDec((unsigned *)acc, value);
}
#endif

template <typename T>
RAJA_INLINE __device__ T hip_atomicDec(T volatile *acc)
{
  return hip_atomic_CAS_oper(acc,
                                      [=] __device__(T a) { return a - 1; });
}


template <typename T>
RAJA_INLINE __device__ T hip_atomicAnd(T volatile *acc, T value)
{
  return hip_atomic_CAS_oper(acc, [=] __device__(T a) {
    return a & value;
  });
}

#if __HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__
// 32-bit signed atomicAnd support by HIP
template <>
RAJA_INLINE __device__ int hip_atomicAnd<int>(int volatile *acc,
                                          int value)
{
  return ::atomicAnd((int *)acc, value);
}


// 32-bit unsigned atomicAnd support by HIP
template <>
RAJA_INLINE __device__ unsigned hip_atomicAnd<unsigned>(unsigned volatile *acc,
                                                    unsigned value)
{
  return ::atomicAnd((unsigned *)acc, value);
}
#endif

// 64-bit unsigned atomicAnd support by HIP sm_35 and later
#if __HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__
template <>
RAJA_INLINE __device__ unsigned long long hip_atomicAnd<unsigned long long>(
    unsigned long long volatile *acc,
    unsigned long long value)
{
  return ::atomicAnd((unsigned long long *)acc, value);
}
#endif

template <typename T>
RAJA_INLINE __device__ T hip_atomicOr(T volatile *acc, T value)
{
  return hip_atomic_CAS_oper(acc, [=] __device__(T a) {
    return a | value;
  });
}

#if __HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__
// 32-bit signed atomicOr support by HIP
template <>
RAJA_INLINE __device__ int hip_atomicOr<int>(int volatile *acc,
                                         int value)
{
  return ::atomicOr((int *)acc, value);
}


// 32-bit unsigned atomicOr support by HIP
template <>
RAJA_INLINE __device__ unsigned hip_atomicOr<unsigned>(unsigned volatile *acc,
                                                   unsigned value)
{
  return ::atomicOr((unsigned *)acc, value);
}
#endif

// 64-bit unsigned atomicOr support by HIP sm_35 and later
#if __HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__
template <>
RAJA_INLINE __device__ unsigned long long hip_atomicOr<unsigned long long>(
    unsigned long long volatile *acc,
    unsigned long long value)
{
  return ::atomicOr((unsigned long long *)acc, value);
}
#endif

template <typename T>
RAJA_INLINE __device__ T hip_atomicXor(T volatile *acc, T value)
{
  return hip_atomic_CAS_oper(acc, [=] __device__(T a) {
    return a ^ value;
  });
}

#if __HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__
// 32-bit signed atomicXor support by HIP
template <>
RAJA_INLINE __device__ int hip_atomicXor<int>(int volatile *acc,
                                          int value)
{
  return ::atomicXor((int *)acc, value);
}


// 32-bit unsigned atomicXor support by HIP
template <>
RAJA_INLINE __device__ unsigned hip_atomicXor<unsigned>(unsigned volatile *acc,
                                                    unsigned value)
{
  return ::atomicXor((unsigned *)acc, value);
}
#endif

#if __HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__
// 64-bit unsigned atomicXor support by HIP sm_35 and later
template <>
RAJA_INLINE __device__ unsigned long long hip_atomicXor<unsigned long long>(
    unsigned long long volatile *acc,
    unsigned long long value)
{
  return ::atomicXor((unsigned long long *)acc, value);
}
#endif

template <typename T>
RAJA_INLINE __device__ T hip_atomicExchange(T volatile *acc, T value)
{
  return hip_atomic_CAS_oper(acc, [=] __device__(T) {
    return value;
  });
}

#if __HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__
template <>
RAJA_INLINE __device__ int hip_atomicExchange<int>(
    int volatile *acc, int value)
{
  return ::atomicExch((int *)acc, value);
}

template <>
RAJA_INLINE __device__ unsigned hip_atomicExchange<unsigned>(
    unsigned volatile *acc, unsigned value)
{
  return ::atomicExch((unsigned *)acc, value);
}
#endif

#if __HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__
template <>
RAJA_INLINE __device__ unsigned long long hip_atomicExchange<unsigned long long>(
    unsigned long long volatile *acc,
    unsigned long long value)
{
  return ::atomicExch((unsigned long long *)acc, value);
}
#endif

#if __HIP_ARCH_HAS_GLOBAL_FLOAT_ATOMIC_EXCH__
template <>
RAJA_INLINE __device__ float hip_atomicExchange<float>(
    float volatile *acc, float value)
{
  return ::atomicExch((float *)acc, value);
}
#endif

template <typename T>
RAJA_INLINE __device__ T hip_atomicCAS(T volatile *acc, T compare, T value)
{
  return hip_atomic_CAS(acc, compare, value);
}

template <>
RAJA_INLINE __device__ int hip_atomicCAS<int>(
    int volatile *acc, int compare, int value)
{
  return ::atomicCAS((int *)acc, compare, value);
}

template <>
RAJA_INLINE __device__ unsigned hip_atomicCAS<unsigned>(
    unsigned volatile *acc, unsigned compare, unsigned value)
{
  return ::atomicCAS((unsigned *)acc, compare, value);
}

template <>
RAJA_INLINE __device__ unsigned long long hip_atomicCAS<unsigned long long>(
    unsigned long long volatile *acc,
    unsigned long long compare,
    unsigned long long value)
{
  return ::atomicCAS((unsigned long long *)acc, compare, value);
}

}  // namespace detail


/*!
 * Hip atomic policy for using hip atomics on the device and
 * the provided host_policy on the host
 */
template<typename host_policy>
struct hip_atomic_explicit{};

/*!
 * Default hip atomic policy uses hip atomics on the device and non-atomics
 * on the host
 */
using hip_atomic = hip_atomic_explicit<loop_atomic>;


/*!
 * Catch-all policy passes off to HIP's builtin atomics.
 *
 * This catch-all will only work for types supported by the compiler.
 * Specialization below can adapt for some unsupported types.
 *
 * These are atomic in hip device code and non-atomic otherwise
 */
RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicAdd(hip_atomic_explicit<host_policy>, T volatile *acc, T value)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicAdd(acc, value);
#else
  return RAJA::atomicAdd(host_policy{}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicSub(hip_atomic_explicit<host_policy>, T volatile *acc, T value)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicSub(acc, value);
#else
  return RAJA::atomicSub(host_policy{}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicMin(hip_atomic_explicit<host_policy>, T volatile *acc, T value)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicMin(acc, value);
#else
  return RAJA::atomicMin(host_policy{}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicMax(hip_atomic_explicit<host_policy>, T volatile *acc, T value)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicMax(acc, value);
#else
  return RAJA::atomicMax(host_policy{}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicInc(hip_atomic_explicit<host_policy>, T volatile *acc, T val)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicInc(acc, val);
#else
  return RAJA::atomicInc(host_policy{}, acc, val);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicInc(hip_atomic_explicit<host_policy>, T volatile *acc)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicInc(acc);
#else
  return RAJA::atomicInc(host_policy{}, acc);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicDec(hip_atomic_explicit<host_policy>, T volatile *acc, T val)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicDec(acc, val);
#else
  return RAJA::atomicDec(host_policy{}, acc, val);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicDec(hip_atomic_explicit<host_policy>, T volatile *acc)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicDec(acc);
#else
  return RAJA::atomicDec(host_policy{}, acc);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicAnd(hip_atomic_explicit<host_policy>, T volatile *acc, T value)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicAnd(acc, value);
#else
  return RAJA::atomicAnd(host_policy{}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicOr(hip_atomic_explicit<host_policy>, T volatile *acc, T value)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicOr(acc, value);
#else
  return RAJA::atomicOr(host_policy{}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicXor(hip_atomic_explicit<host_policy>, T volatile *acc, T value)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicXor(acc, value);
#else
  return RAJA::atomicXor(host_policy{}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicExchange(hip_atomic_explicit<host_policy>, T volatile *acc, T value)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicExchange(acc, value);
#else
  return RAJA::atomicExchange(host_policy{}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicCAS(hip_atomic_explicit<host_policy>, T volatile *acc, T compare, T value)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicCAS(acc, compare, value);
#else
  return RAJA::atomicCAS(host_policy{}, acc, compare, value);
#endif
}

}  // namespace RAJA


#endif  // RAJA_ENABLE_HIP
#endif  // guard
