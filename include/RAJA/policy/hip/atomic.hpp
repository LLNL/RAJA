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
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
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

#include "RAJA/util/camp_aliases.hpp"
#include "RAJA/util/concepts.hpp"
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


template < typename T, typename TypeList >
struct is_any_of;

template < typename T, typename... Types >
struct is_any_of<T, list<Types...>>
  : concepts::any_of<camp::is_same<T, Types>...>
{};

template < typename T, typename TypeList >
using enable_if_is_any_of = std::enable_if_t<is_any_of<T, TypeList>::value, T>;

template < typename T, typename TypeList >
using enable_if_is_none_of = std::enable_if_t<concepts::negate<is_any_of<T, TypeList>>::value, T>;


using hip_atomicCommon_builtin_types = list<
      int
     ,unsigned int
     ,unsigned long long
    >;


using hip_atomicAdd_builtin_types = list<
      int
     ,unsigned int
     ,unsigned long long
     ,float
#ifdef RAJA_ENABLE_HIP_DOUBLE_ATOMICADD
     ,double
#endif
    >;

/*!
 * List of types where HIP builtin atomics are used to implement atomicSub.
 */
using hip_atomicSub_types = list<
      int
     ,unsigned int
     ,float
#ifdef RAJA_ENABLE_HIP_DOUBLE_ATOMICADD
     ,double
#endif
    >;

using hip_atomicSub_builtin_types = list<
      int
     ,unsigned int
    >;

/*!
 * List of types where HIP builtin atomicAdd is used to implement atomicSub.
 *
 * Avoid multiple definition errors by including the previous list type here
 * to ensure these lists have different types.
 */
using hip_atomicSub_via_Add_builtin_types = list<
      float
#ifdef RAJA_ENABLE_HIP_DOUBLE_ATOMICADD
     ,double
#endif
    >;

using hip_atomicMin_builtin_types = hip_atomicCommon_builtin_types;

using hip_atomicMax_builtin_types = hip_atomicCommon_builtin_types;

using hip_atomicIncReset_builtin_types = list<
      unsigned int
    >;

using hip_atomicInc_builtin_types = list< >;

using hip_atomicDecReset_builtin_types = list<
      unsigned int
    >;

using hip_atomicDec_builtin_types = list< >;

using hip_atomicAnd_builtin_types = hip_atomicCommon_builtin_types;

using hip_atomicOr_builtin_types = hip_atomicCommon_builtin_types;

using hip_atomicXor_builtin_types = hip_atomicCommon_builtin_types;

using hip_atomicExch_builtin_types = list<
      int
     ,unsigned int
     ,unsigned long long
     ,float
    >;

using hip_atomicCAS_builtin_types = hip_atomicCommon_builtin_types;


template <typename T, enable_if_is_none_of<T, hip_atomicAdd_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicAdd(T volatile *acc, T value)
{
  return hip_atomic_CAS_oper(acc, [=] __device__(T a) {
    return a + value;
  });
}

template <typename T, enable_if_is_any_of<T, hip_atomicAdd_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicAdd(T volatile *acc, T value)
{
  return ::atomicAdd((T *)acc, value);
}


template <typename T, enable_if_is_none_of<T, hip_atomicSub_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicSub(T volatile *acc, T value)
{
  return hip_atomic_CAS_oper(acc, [=] __device__(T a) {
    return a - value;
  });
}

/*!
 * HIP atomicSub builtin implementation.
 */
template <typename T, enable_if_is_any_of<T, hip_atomicSub_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicSub(T volatile *acc, T value)
{
  return ::atomicSub((T *)acc, value);
}

/*!
 * HIP atomicSub via atomicAdd builtin implementation.
 */
template <typename T, enable_if_is_any_of<T, hip_atomicSub_via_Add_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicSub(T volatile *acc, T value)
{
  return ::atomicAdd((T *)acc, -value);
}


template <typename T, enable_if_is_none_of<T, hip_atomicMin_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicMin(T volatile *acc, T value)
{
  return hip_atomic_CAS_oper(acc, [=] __device__(T a) {
    return value < a ? value : a;
  });
}

template <typename T, enable_if_is_any_of<T, hip_atomicMin_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicMin(T volatile *acc, T value)
{
  return ::atomicMin((T *)acc, value);
}


template <typename T, enable_if_is_none_of<T, hip_atomicMax_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicMax(T volatile *acc, T value)
{
  return hip_atomic_CAS_oper(acc, [=] __device__(T a) {
    return value > a ? value : a;
  });
}

template <typename T, enable_if_is_any_of<T, hip_atomicMax_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicMax(T volatile *acc, T value)
{
  return ::atomicMax((T *)acc, value);
}


template <typename T, enable_if_is_none_of<T, hip_atomicIncReset_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicInc(T volatile *acc, T val)
{
  return hip_atomic_CAS_oper(acc, [=] __device__(T old) {
    return ((old >= val) ? (T)0 : (old + (T)1));
  });
}

template <typename T, enable_if_is_any_of<T, hip_atomicIncReset_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicInc(T volatile *acc, T val)
{
  return ::atomicInc((T *)acc, val);
}


template <typename T, enable_if_is_none_of<T, hip_atomicInc_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicInc(T volatile *acc)
{
  return hip_atomicAdd(acc, (T)1);
}

template <typename T, enable_if_is_any_of<T, hip_atomicInc_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicInc(T volatile *acc)
{
  return ::atomicInc((T *)acc);
}


template <typename T, enable_if_is_none_of<T, hip_atomicDecReset_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicDec(T volatile *acc, T val)
{
  // See:
  // http://docs.nvidia.com/hip/hip-c-programming-guide/index.html#atomicdec
  return hip_atomic_CAS_oper(acc, [=] __device__(T old) {
    return (((old == (T)0) | (old > val)) ? val : (old - (T)1));
  });
}

template <typename T, enable_if_is_any_of<T, hip_atomicDecReset_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicDec(T volatile *acc, T val)
{
  return ::atomicDec((T *)acc, val);
}


template <typename T, enable_if_is_none_of<T, hip_atomicDec_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicDec(T volatile *acc)
{
  return hip_atomicSub(acc, (T)1);
}

template <typename T, enable_if_is_any_of<T, hip_atomicDec_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicDec(T volatile *acc)
{
  return ::atomicDec((T *)acc);
}


template <typename T, enable_if_is_none_of<T, hip_atomicAnd_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicAnd(T volatile *acc, T val)
{
  return hip_atomic_CAS_oper(acc, [=] __device__(T a) {
    return a & val;
  });
}

template <typename T, enable_if_is_any_of<T, hip_atomicAnd_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicAnd(T volatile *acc, T val)
{
  return ::atomicAnd((T *)acc, val);
}


template <typename T, enable_if_is_none_of<T, hip_atomicOr_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicOr(T volatile *acc, T val)
{
  return hip_atomic_CAS_oper(acc, [=] __device__(T a) {
    return a | val;
  });
}

template <typename T, enable_if_is_any_of<T, hip_atomicOr_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicOr(T volatile *acc, T val)
{
  return ::atomicOr((T *)acc, val);
}


template <typename T, enable_if_is_none_of<T, hip_atomicXor_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicXor(T volatile *acc, T val)
{
  return hip_atomic_CAS_oper(acc, [=] __device__(T a) {
    return a ^ val;
  });
}

template <typename T, enable_if_is_any_of<T, hip_atomicXor_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicXor(T volatile *acc, T val)
{
  return ::atomicXor((T *)acc, val);
}


template <typename T, enable_if_is_none_of<T, hip_atomicExch_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicExchange(T volatile *acc, T val)
{
  return hip_atomic_CAS_oper(acc, [=] __device__(T) {
    return val;
  });
}

template <typename T, enable_if_is_any_of<T, hip_atomicExch_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicExchange(T volatile *acc, T val)
{
  return ::atomicExch((T *)acc, val);
}


template <typename T, enable_if_is_none_of<T, hip_atomicCAS_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicCAS(T volatile *acc, T compare, T val)
{
  return hip_atomic_CAS(acc, compare, val);
}

template <typename T, enable_if_is_any_of<T, hip_atomicCAS_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicCAS( T volatile *acc, T compare, T val)
{
  return ::atomicCAS((T *)acc, compare, val);
}

}  // namespace detail


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
