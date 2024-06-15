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

// TODO: When we can use if constexpr in C++17, this file can be cleaned up

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
     ,unsigned long long
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
      unsigned long long
     ,float
#ifdef RAJA_ENABLE_HIP_DOUBLE_ATOMICADD
     ,double
#endif
    >;

using hip_atomicMin_builtin_types = hip_atomicCommon_builtin_types;

using hip_atomicMax_builtin_types = hip_atomicCommon_builtin_types;

using hip_atomicIncReset_builtin_types = list< >;

using hip_atomicInc_builtin_types = list< >;

using hip_atomicDecReset_builtin_types = list< >;

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

static_assert(sizeof(unsigned int) != sizeof(unsigned long long int),
              "Current implementation of Hip atomicCAS expects unsigned int and unsigned long long int to be different sizes (e.g. 4 vs. 8 bytes)");

/*!
 * Provides a member called "type" when the size matches the size of a
 * type that can be passed to Hip's atomicCAS function. Otherwise, a
 * compile-time assert is triggered.
 */
template <std::size_t BYTES>
struct hip_atomicCAS_reinterpret_cast_impl {
  static_assert(false, "Unsupported type for atomicCAS");
};

/*!
 * Provides a member called "type" when the size matches the size of a
 * type that can be passed to Hip's atomicCAS function. Otherwise, a
 * compile-time assert is triggered. Specialization for the size of
 * unsigned int (usually 4 bytes).
 */
template <>
struct hip_atomicCAS_reinterpret_cast_impl<sizeof(unsigned int)> {
  using type = unsigned int;
};

/*!
 * Provides a member called "type" when the size matches the size of a
 * type that can be passed to Hip's atomicCAS function. Otherwise, a
 * compile-time assert is triggered. Specialization for the size of
 * unsigned long long int (usually 8 bytes).
 */
template <>
struct hip_atomicCAS_reinterpret_cast_impl<sizeof(unsigned long long int)> {
  using type = unsigned long long int;
};

/*!
 * Reinterprets the given type as a type that can be passed to Hip's
 * atomicCAS function.
 */
template <class T>
struct hip_atomicCAS_reinterpret_cast {
  using type = typename hip_atomicCAS_reinterpret_cast_impl<sizeof(T)>::type;
};

/*!
 * Reinterprets the given type as a type that can be passed to Hip's
 * atomicCAS function. This is a specialization for type int.
 */
template <>
struct hip_atomicCAS_reinterpret_cast<int> {
  using type = int;
};

/*!
 * Alias for hip_atomicCAS_reinterpret_cast<T>::type.
 */
template <class T>
using hip_atomicCAS_reinterpret_cast_t =
    typename hip_atomicCAS_reinterpret_cast<T>::type;

/*
 * Performs an atomic exchange. Stores the new value in the given address
 * and returns the old value.
 */
template <typename T,
          std::enable_if_t<std::is_same<T, int>::value ||
                           std::is_same<T, unsigned int>::value ||
                           std::is_same<T, unsigned long long int>::value ||
                           std::is_same<T, float>::value, bool> = true>
RAJA_INLINE __device__ T hip_atomicExchange(T *acc, T value)
{
  return ::atomicExch(acc, value);
}

template <typename T,
          std::enable_if_t<!std::is_same<T, int>::value &&
                           !std::is_same<T, unsigned int>::value &&
                           !std::is_same<T, unsigned long long int>::value &&
                           !std::is_same<T, float>::value &&
                           sizeof(T) == sizeof(unsigned int), bool> = true>
RAJA_INLINE __device__ T hip_atomicExchange(T *acc, T value)
{
  return hip_atomicExchange(reinterpret_cast<unsigned int*>(acc),
                            RAJA::util::reinterp_A_as_B<T, unsigned int>(value));
}

template <typename T,
          std::enable_if_t<!std::is_same<T, int>::value &&
                           !std::is_same<T, unsigned int>::value &&
                           !std::is_same<T, unsigned long long int>::value &&
                           !std::is_same<T, float>::value &&
                           sizeof(T) == sizeof(unsigned long long int), bool> = true>
RAJA_INLINE __device__ T hip_atomicExchange(T *acc, T value)
{
  return hip_atomicExchange(reinterpret_cast<unsigned long long int*>(acc),
                            RAJA::util::reinterp_A_as_B<T, unsigned long long int>(value));
}


#if defined(__has_builtin) && __has_builtin(__hip_atomic_load)

template <typename T,
          std::enable_if_t<std::is_arithmetic<T>::value ||
                           std::is_enum<T>::value, bool> = true>
RAJA_INLINE __device__ T hip_atomicLoad(T *acc)
{
  return __hip_atomic_load(acc, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
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
          std::enable_if_t<std::is_same<T, int>::value ||
                           std::is_same<T, unsigned int>::value ||
                           std::is_same<T, unsigned long long int>::value, bool> = true>
RAJA_INLINE __device__ T hip_atomicLoad(T *acc)
{
  return ::atomicOr(acc, 0);
}

template <typename T,
          std::enable_if_t<!std::is_same<T, int>::value &&
                           !std::is_same<T, unsigned int>::value &&
                           !std::is_same<T, unsigned long long int>::value &&
                           sizeof(T) == sizeof(unsigned int), bool> = true>
RAJA_INLINE __device__ T hip_atomicLoad(T *acc)
{
  return hip_atomicLoad(reinterpret_cast<unsigned int*>(acc));
}

template <typename T,
          std::enable_if_t<!std::is_same<T, int>::value &&
                           !std::is_same<T, unsigned int>::value &&
                           !std::is_same<T, unsigned long long int>::value &&
                           sizeof(T) == sizeof(unsigned long long int), bool> = true>
RAJA_INLINE __device__ T hip_atomicLoad(T *acc)
{
  return hip_atomicLoad(reinterpret_cast<unsigned long long int*>(acc));
}

#endif



#if defined(__has_builtin) && __has_builtin(__hip_atomic_store)

template <typename T,
          std::enable_if_t<std::is_arithmetic<T>::value ||
                           std::is_enum<T>::value, bool> = true>
RAJA_INLINE __device__ void hip_atomicStore(T *acc, T value)
{
  return __hip_atomic_store(acc, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

#if defined(UINT8_MAX)

template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint8_t), bool> = true>
RAJA_INLINE __device__ void hip_atomicStore(T *acc, T value)
{
  hip_atomicStore(reinterpret_cast<uint8_t *>(acc),
                  RAJA::util::reinterp_A_as_B<T, uint8_t>(value));
}

#else

template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(unsigned char) == 1 &&
                           sizeof(T) == 1, bool> = true>
RAJA_INLINE __device__ void hip_atomicStore(T *acc, T value)
{
  hip_atomicStore(reinterpret_cast<unsigned char *>(acc),
                  RAJA::util::reinterp_A_as_B<T, unsigned char>(value));
}

#endif

#if defined(UINT16_MAX)

template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint16_t), bool> = true>
RAJA_INLINE __device__ void hip_atomicStore(T *acc, T value)
{
  hip_atomicStore(reinterpret_cast<uint16_t *>(acc),
                  RAJA::util::reinterp_A_as_B<T, uint16_t>(value));
}

#else

template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(unsigned short int) == 2 &&
                           sizeof(T) == 2, bool> = true>
RAJA_INLINE __device__ void hip_atomicStore(T *acc, T value)
{
  hip_atomicStore(reinterpret_cast<unsigned short int *>(acc),
                  RAJA::util::reinterp_A_as_B<T, unsigned short int>(value));
}

#endif

#if defined(UINT32_MAX)

template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint32_t), bool> = true>
RAJA_INLINE __device__ void hip_atomicStore(T *acc, T value)
{
  hip_atomicStore(reinterpret_cast<uint32_t *>(acc),
                  RAJA::util::reinterp_A_as_B<T, uint32_t>(value));
}

#else

template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(unsigned int) == 4 &&
                           sizeof(T) == 4, bool> = true>
RAJA_INLINE __device__ void hip_atomicStore(T *acc, T value)
{
  hip_atomicStore(reinterpret_cast<unsigned int *>(acc),
                  RAJA::util::reinterp_A_as_B<T, unsigned int>(value));
}

#endif

#if defined(UINT64_MAX)

template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint64_t), bool> = true>
RAJA_INLINE __device__ void hip_atomicStore(T *acc, T value)
{
  hip_atomicStore(reinterpret_cast<uint64_t *>(acc),
                  RAJA::util::reinterp_A_as_B<T, uint64_t>(value));
}

#else

template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(unsigned long long int) == 8 &&
                           sizeof(T) == 8, bool> = true>
RAJA_INLINE __device__ void hip_atomicStore(T *acc, T value)
{
  hip_atomicStore(reinterpret_cast<unsigned long long int *>(acc),
                  RAJA::util::reinterp_A_as_B<T, unsigned long long int>(value));
}

#endif

#else

template <typename T>
RAJA_INLINE __device__ void hip_atomicStore(T *acc, T value)
{
  hip_atomicExchange(acc, value);
}

#endif



/*!
 * Hip atomicCAS using builtins
 *
 * Returns the old value in memory before this operation.
 */
template <typename T>
RAJA_INLINE __device__ T hip_atomicCAS(T *acc, T compare, T value)
{
  using hip_atomicCAS_type = hip_atomicCAS_reinterpret_cast_t<T>;

  return RAJA::util::reinterp_A_as_B<hip_atomicCAS_type, T>(
      ::atomicCAS(
          reinterpret_cast<hip_atomicCAS_type *>(acc),
          RAJA::util::reinterp_A_as_B<T, hip_atomicCAS_type>(compare),
          RAJA::util::reinterp_A_as_B<T, hip_atomicCAS_type>(value)));
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
    old = hip_atomicCAS(acc, expected, oper(expected));
  } while (RAJA::util::reinterp_A_as_B<T, hip_atomicCAS_type>(old) !=
           RAJA::util::reinterp_A_as_B<T, hip_atomicCAS_type>(expected));

  // The while conditional must use the underlying integral type to avoid
  // cases like NaNs, which will never be equal.

  return old;
}


template <typename T, enable_if_is_none_of<T, hip_atomicAdd_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicAdd(T *acc, T value)
{
  return hip_atomicCAS(acc, [value] __device__(T a) {
    return a + value;
  });
}

template <typename T, enable_if_is_any_of<T, hip_atomicAdd_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicAdd(T *acc, T value)
{
  return ::atomicAdd(acc, value);
}


template <typename T, enable_if_is_none_of<T, hip_atomicSub_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicSub(T *acc, T value)
{
  return hip_atomicCAS(acc, [value] __device__(T a) {
    return a - value;
  });
}

/*!
 * HIP atomicSub builtin implementation.
 */
template <typename T, enable_if_is_any_of<T, hip_atomicSub_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicSub(T *acc, T value)
{
  return ::atomicSub(acc, value);
}

/*!
 * HIP atomicSub via atomicAdd builtin implementation.
 */
template <typename T, enable_if_is_any_of<T, hip_atomicSub_via_Add_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicSub(T *acc, T value)
{
  return ::atomicAdd(acc, -value);
}


template <typename T, enable_if_is_none_of<T, hip_atomicMin_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicMin(T *acc, T value)
{
  return hip_atomicCAS(acc, [value] __device__(T a) {
    return value < a ? value : a;
  });
}

template <typename T, enable_if_is_any_of<T, hip_atomicMin_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicMin(T *acc, T value)
{
  return ::atomicMin(acc, value);
}


template <typename T, enable_if_is_none_of<T, hip_atomicMax_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicMax(T *acc, T value)
{
  return hip_atomicCAS(acc, [value] __device__(T a) {
    return a < value ? value : a;
  });
}

template <typename T, enable_if_is_any_of<T, hip_atomicMax_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicMax(T *acc, T value)
{
  return ::atomicMax(acc, value);
}


template <typename T, enable_if_is_none_of<T, hip_atomicIncReset_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicInc(T *acc, T value)
{
  return hip_atomicCAS(acc, [value] __device__(T old) {
    return value <= old ? static_cast<T>(0) : old + static_cast<T>(1);
  });
}

template <typename T, enable_if_is_any_of<T, hip_atomicIncReset_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicInc(T *acc, T value)
{
  return ::atomicInc(acc, value);
}


template <typename T, enable_if_is_none_of<T, hip_atomicInc_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicInc(T *acc)
{
  return hip_atomicAdd(acc, static_cast<T>(1));
}

template <typename T, enable_if_is_any_of<T, hip_atomicInc_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicInc(T *acc)
{
  return ::atomicInc(acc);
}


template <typename T, enable_if_is_none_of<T, hip_atomicDecReset_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicDec(T *acc, T value)
{
  return hip_atomicCAS(acc, [value] __device__(T old) {
    return old == static_cast<T>(0) || value < old ? value : old - static_cast<T>(1);
  });
}

template <typename T, enable_if_is_any_of<T, hip_atomicDecReset_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicDec(T *acc, T value)
{
  return ::atomicDec(acc, value);
}


template <typename T, enable_if_is_none_of<T, hip_atomicDec_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicDec(T *acc)
{
  return hip_atomicSub(acc, static_cast<T>(1));
}

template <typename T, enable_if_is_any_of<T, hip_atomicDec_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicDec(T *acc)
{
  return ::atomicDec(acc);
}


template <typename T, enable_if_is_none_of<T, hip_atomicAnd_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicAnd(T *acc, T value)
{
  return hip_atomicCAS(acc, [value] __device__(T a) {
    return a & value;
  });
}

template <typename T, enable_if_is_any_of<T, hip_atomicAnd_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicAnd(T *acc, T value)
{
  return ::atomicAnd(acc, value);
}


template <typename T, enable_if_is_none_of<T, hip_atomicOr_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicOr(T *acc, T value)
{
  return hip_atomicCAS(acc, [value] __device__(T a) {
    return a | value;
  });
}

template <typename T, enable_if_is_any_of<T, hip_atomicOr_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicOr(T *acc, T value)
{
  return ::atomicOr(acc, value);
}


template <typename T, enable_if_is_none_of<T, hip_atomicXor_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicXor(T *acc, T value)
{
  return hip_atomicCAS(acc, [value] __device__(T a) {
    return a ^ value;
  });
}

template <typename T, enable_if_is_any_of<T, hip_atomicXor_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicXor(T *acc, T value)
{
  return ::atomicXor(acc, value);
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
atomicLoad(hip_atomic_explicit<host_policy>, T *acc)
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
atomicStore(hip_atomic_explicit<host_policy>, T *acc, T value)
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
atomicAdd(hip_atomic_explicit<host_policy>, T *acc, T value)
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
atomicSub(hip_atomic_explicit<host_policy>, T *acc, T value)
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
atomicMin(hip_atomic_explicit<host_policy>, T *acc, T value)
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
atomicMax(hip_atomic_explicit<host_policy>, T *acc, T value)
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
atomicInc(hip_atomic_explicit<host_policy>, T *acc, T value)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicInc(acc, value);
#else
  return RAJA::atomicInc(host_policy{}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicInc(hip_atomic_explicit<host_policy>, T *acc)
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
atomicDec(hip_atomic_explicit<host_policy>, T *acc, T value)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return detail::hip_atomicDec(acc, value);
#else
  return RAJA::atomicDec(host_policy{}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicDec(hip_atomic_explicit<host_policy>, T *acc)
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
atomicAnd(hip_atomic_explicit<host_policy>, T *acc, T value)
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
atomicOr(hip_atomic_explicit<host_policy>, T *acc, T value)
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
atomicXor(hip_atomic_explicit<host_policy>, T *acc, T value)
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
atomicExchange(hip_atomic_explicit<host_policy>, T *acc, T value)
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
atomicCAS(hip_atomic_explicit<host_policy>, T *acc, T compare, T value)
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
