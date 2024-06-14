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
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_hip_atomic_HPP
#define RAJA_policy_hip_atomic_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include "hip/hip_runtime.h"

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

/*!
 * Reinterprets the given type as a type that can be passed to Hip's
 * atomicCAS function. This specialization handles types that can be
 * passed directly.
 */
template <class T,
          std::enable_if_t<!std::is_same<T, int>::value &&
                           !std::is_same<T, unsigned int>::value &&
                           !std::is_same<T, unsigned long long int>::value &&
                           sizeof(T) != sizeof(unsigned int) &&
                           sizeof(T) != sizeof(unsigned long long int)> = true>
struct hip_atomicCAS_reinterpret_cast {
  static_assert("atomicCAS is not supported for given type");
};

/*!
 * Reinterprets the given type as a type that can be passed to Hip's
 * atomicCAS function. This specialization handles types that can be
 * passed directly.
 */
template <class T,
          enable_if_is_any_of<T, hip_atomicCAS_builtin_types>* = nullptr>
struct hip_atomicCAS_reinterpret_cast {
  using type = T;
};

/*!
 * Reinterprets the given type as a type that can be passed to Hip's
 * atomicCAS function. This specialization handles types that can be
 * reinterpreted as an unsigned int.
 */
template <class T,
          std::enable_if_t<!std::is_same<T, int>::value &&
                           !std::is_same<T, unsigned int>::value &&
                           sizeof(T) == sizeof(unsigned int)> = true>
struct hip_atomicCAS_reinterpret_cast {
  using type = unsigned int;
};

/*!
 * Reinterprets the given type as a type that can be passed to Hip's
 * atomicCAS function. This specialization handles types that can be
 * reinterpreted as an unsigned long long int.
 */
template <class T,
          std::enable_if_t<!std::is_same<T, unsigned long long int>::value &&
                           sizeof(T) == sizeof(unsigned long long int)> = true>
struct hip_atomicCAS_reinterpret_cast {
  using type = unsigned long long int;
};

/*!
 * Alias for hip_atomicCAS_reinterpret_cast<T>::type.
 */
template <class T>
using hip_atomicCAS_reinterpret_cast_t =
    typename hip_atomicCAS_reinterpret_cast::type;

#if defined(__has_builtin) && __has_builtin(__hip_atomic_load)

template <typename T,
          std::enable_if_t<std::is_arithmetic<T>::value ||
                           std::is_enum<T>::value, bool> = true>
RAJA_INLINE __device__ T hip_atomicLoad(T *acc)
{
  return __hip_atomic_load((T *)acc, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

#if defined(UINT8_MAX)

template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint8_t), bool> = true>
RAJA_INLINE __device__ T hip_atomicLoad(T *acc)
{
  return hip_atomicLoad(reinterpret_cast<uint8_t *>(acc));
}

#else

template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(unsigned char) == 1 &&
                           sizeof(T) == 1, bool> = true>
RAJA_INLINE __device__ T hip_atomicLoad(T *acc)
{
  return hip_atomicLoad(reinterpret_cast<unsigned char *>(acc));
}

#endif

#if defined(UINT16_MAX)

template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint16_t), bool> = true>
RAJA_INLINE __device__ T hip_atomicLoad(T *acc)
{
  return hip_atomicLoad(reinterpret_cast<uint16_t *>(acc));
}

#else

template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(unsigned short int) == 2 &&
                           sizeof(T) == 2, bool> = true>
RAJA_INLINE __device__ T hip_atomicLoad(T *acc)
{
  return hip_atomicLoad(reinterpret_cast<unsigned short int *>(acc));
}

#endif

#if defined(UINT32_MAX)

template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint32_t), bool> = true>
RAJA_INLINE __device__ T hip_atomicLoad(T *acc)
{
  return hip_atomicLoad(reinterpret_cast<uint32_t *>(acc));
}

#else

template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(unsigned int) == 4 &&
                           sizeof(T) == 4, bool> = true>
RAJA_INLINE __device__ T hip_atomicLoad(T *acc)
{
  return hip_atomicLoad(reinterpret_cast<unsigned int *>(acc));
}

#endif

#if defined(UINT64_MAX)

template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint64_t), bool> = true>
RAJA_INLINE __device__ T hip_atomicLoad(T *acc)
{
  return hip_atomicLoad(reinterpret_cast<uint64_t *>(acc));
}

#else

template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(unsigned long long int) == 8 &&
                           sizeof(T) == 8, bool> = true>
RAJA_INLINE __device__ T hip_atomicLoad(T *acc)
{
  return hip_atomicLoad(reinterpret_cast<unsigned long long int *>(acc));
}

#endif

#else

template <typename T,
          enable_if_is_any_of<T, hip_atomicOr_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicLoad(T *acc)
{
  return ::atomicOr(acc, 0);
}

template <typename T,
          enable_if_is_none_of<T, hip_atomicOr_builtin_types>* = nullptr,
          std::enable_if_t<sizeof(T) == sizeof(unsigned int)> = true>
RAJA_INLINE __device__ T hip_atomicLoad(T *acc)
{
  return hip_atomicLoad(reinterpret_cast<unsigned int*>(acc));
}

template <typename T,
          enable_if_is_none_of<T, hip_atomicOr_builtin_types>* = nullptr,
          std::enable_if_t<sizeof(T) == sizeof(unsigned long long int)> = true>
RAJA_INLINE __device__ T hip_atomicLoad(T *acc)
{
  return hip_atomicLoad(reinterpret_cast<unsigned long long int*>(acc));
}

#endif

/*!
 * Hip atomicCAS using builtins
 *
 * Returns the old value in memory before this operation.
 */
template <typename T>
RAJA_INLINE __device__ T hip_atomicCAS(T *acc, T compare, T val)
{
  using hip_atomicCAS_type = hip_atomicCAS_reinterpret_cast_t<T>;

  return RAJA::util::reinterp_A_as_B<hip_atomicCAS_type, T>(
      ::atomicCAS(
          reinterpret_cast<hip_atomicCAS_type *>(acc),
          RAJA::util::reinterp_A_as_B<T, hip_atomicCAS_type>(compare),
          RAJA::util::reinterp_A_as_B<T, hip_atomicCAS_type>(val)));
}

/*!
* Generic impementation of any 32-bit or 64-bit atomic operator.
* Implementation uses the existing Hip supplied unsigned int 32-bit CAS
* operator or unsigned long long int 64-bit CAS operator. Returns the
* OLD value that was replaced by the result of this operation.
*/
template <typename T, typename OPER>
RAJA_INLINE __device__ T hip_atomicCAS(T *acc, OPER &&oper)
{
  using hip_atomicCAS_type = hip_atomicCAS_reinterpret_cast_t<T>;

  T old = hip_atomicLoad(acc);
  T expected;

  do {
    expected = old;
    old = hip_atomicCas(acc, expected, oper(expected));
  } while (RAJA::util::reinterp_A_as_B<T, hip_atomicCAS_type>(old) !=
           RAJA::util::reinterp_A_as_B<T, hip_atomicCAS_type>(expected));

  // The while conditional must use the underlying integral type to avoid
  // cases like NaNs, which will never be equal.

  return old;
}

template <typename T
#if defined(__has_builtin) && __has_builtin(__hip_atomic_store)
          , std::enable_if_t<!(std::is_arithmetic<T>::value ||
                               std::is_enum<T>::value), bool> = true
#endif
         >
RAJA_INLINE __device__ void hip_atomicStore(T volatile *acc, T val)
{
  hip_atomicCAS(acc, [=] __device__(T) {
    return val;
  });
}

#if defined(__has_builtin) && __has_builtin(__hip_atomic_store)
template <typename T,
          std::enable_if_t<std::is_arithmetic<T>::value ||
                           std::is_enum<T>::value, bool> = true>
RAJA_INLINE __device__ void hip_atomicStore(T volatile *acc, T val)
{
  __hip_atomic_store((T *)acc, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
#endif


template <typename T, enable_if_is_none_of<T, hip_atomicAdd_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicAdd(T volatile *acc, T value)
{
  return hip_atomicCAS(acc, [=] __device__(T a) {
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
  return hip_atomicCAS(acc, [=] __device__(T a) {
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
  return hip_atomicCAS(acc, [=] __device__(T a) {
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
  return hip_atomicCAS(acc, [=] __device__(T a) {
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
  return hip_atomicCAS(acc, [=] __device__(T old) {
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
  return hip_atomicCAS(acc, [=] __device__(T old) {
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
  return hip_atomicCAS(acc, [=] __device__(T a) {
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
  return hip_atomicCAS(acc, [=] __device__(T a) {
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
  return hip_atomicCAS(acc, [=] __device__(T a) {
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
  return hip_atomicCAS(acc, [=] __device__(T) {
    return val;
  });
}

template <typename T, enable_if_is_any_of<T, hip_atomicExch_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicExchange(T volatile *acc, T val)
{
  return ::atomicExch((T *)acc, val);
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
atomicLoad(hip_atomic_explicit<host_policy>, T volatile *acc)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicLoad(acc);
#else
  return RAJA::atomicLoad(host_policy{}, acc);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE void
atomicStore(hip_atomic_explicit<host_policy>, T volatile *acc, T value)
{
#if defined(__HIP_DEVICE_COMPILE__)
  detail::hip_atomicStore(acc, value);
#else
  RAJA::atomicStore(host_policy{}, acc, value);
#endif
}

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
