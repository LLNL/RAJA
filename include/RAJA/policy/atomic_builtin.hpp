/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining automatic and builtin atomic operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_atomic_builtin_HPP
#define RAJA_policy_atomic_builtin_HPP

#include "RAJA/config.hpp"

#include <cstdint>

#if defined(RAJA_COMPILER_MSVC) || (defined(_WIN32) && defined(__INTEL_COMPILER))
#include <intrin.h>
#endif

#include "RAJA/util/TypeConvert.hpp"
#include "RAJA/util/macros.hpp"


#if defined(RAJA_ENABLE_HIP)
#define RAJA_DEVICE_HIP RAJA_HOST_DEVICE
#else
#define RAJA_DEVICE_HIP
#endif

namespace RAJA
{


//! Atomic policy that uses the compilers builtin __atomic_XXX routines
struct builtin_atomic {
};


namespace detail {


#if defined(RAJA_COMPILER_MSVC) || (defined(_WIN32) && defined(__INTEL_COMPILER))


/*!
 * Type trait for determining if the operator should be implemented
 * using an intrinsic
 */
template <typename T>
struct builtin_useIntrinsic {
  static constexpr bool value =
    std::is_same<T, char>::value ||
    std::is_same<T, short>::value ||
    std::is_same<T, long>::value ||
    std::is_same<T, long long>::value;
};


/*!
 * Type trait for determining if the operator should be implemented
 * by reinterpreting inputs to types that intrinsics support
 */
template <typename T>
struct builtin_useReinterpret {
  static constexpr bool value =
    !std::is_integral<T>::value &&
    !std::is_enum<T>::value &&
    (sizeof(T) == 1 ||
     sizeof(T) == 2 ||
     sizeof(T) == 4 ||
     sizeof(T) == 8);

  using type =
    std::conditional_t<sizeof(T) == 1, char,
    std::conditional_t<sizeof(T) == 2, short,
    std::conditional_t<sizeof(T) == 4, long, long long>>>;
};


/*!
 * Atomic load using intrinsics
 */
RAJA_INLINE char builtin_atomicLoad(char *acc)
{
  return _InterlockedOr8(acc, static_cast<char>(0));
}

RAJA_INLINE short builtin_atomicLoad(short *acc)
{
  return _InterlockedOr16(acc, static_cast<short>(0));
}

RAJA_INLINE long builtin_atomicLoad(long *acc)
{
  return _InterlockedOr(acc, static_cast<long>(0));
}

RAJA_INLINE long long builtin_atomicLoad(long long *acc)
{
  return _InterlockedOr64(acc, static_cast<long long>(0));
}


/*!
 * Atomic exchange using intrinsics
 */
RAJA_INLINE char builtin_atomicExchange(char *acc, char value)
{
  return _InterlockedExchange8(acc, value);
}

RAJA_INLINE short builtin_atomicExchange(short *acc, short value)
{
  return _InterlockedExchange16(acc, value);
}

RAJA_INLINE long builtin_atomicExchange(long *acc, long value)
{
  return _InterlockedExchange(acc, value);
}

RAJA_INLINE long long builtin_atomicExchange(long long *acc, long long value)
{
  return _InterlockedExchange64(acc, value);
}


/*!
 * Atomic store (implemented using atomic exchange)
 */
template <typename T>
RAJA_INLINE void builtin_atomicStore(T *acc, T value)
{
  builtin_atomicExchange(acc, value);
}


/*!
 * Atomic compare and swap using intrinsics
 */
RAJA_INLINE char builtin_atomicCAS(char *acc, char compare, char value)
{
  return _InterlockedCompareExchange8(acc, value, compare);
}

RAJA_INLINE short builtin_atomicCAS(short *acc, short compare, short value)
{
  return _InterlockedCompareExchange16(acc, value, compare);
}

RAJA_INLINE long builtin_atomicCAS(long *acc, long compare, long value)
{
  return _InterlockedCompareExchange(acc, value, compare);
}

RAJA_INLINE long long builtin_atomicCAS(long long *acc, long long compare, long long value)
{
  return _InterlockedCompareExchange64(acc, value, compare);
}


#else  // RAJA_COMPILER_MSVC


/*!
 * Type trait for determining if the operator should be implemented
 * using an intrinsic
 */
template <typename T>
struct builtin_useIntrinsic {
  static constexpr bool value =
    (std::is_integral<T>::value || std::is_enum<T>::value) &&
    (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);
};


/*!
 * Type trait for determining if the operator should be implemented
 * by reinterpreting inputs to types that intrinsics support
 */
template <typename T>
struct builtin_useReinterpret {
  static constexpr bool value =
    !std::is_integral<T>::value &&
    !std::is_enum<T>::value &&
    ((sizeof(T) == 1
#if !defined(UINT8_MAX)
      && sizeof(unsigned char) == 1
#endif
     ) ||
     (sizeof(T) == 2
#if !defined(UINT16_MAX)
      && sizeof(unsigned short) == 2
#endif
     ) ||
     (sizeof(T) == 4
#if !defined(UINT32_MAX)
      && sizeof(unsigned int) == 4
#endif
     ) ||
     (sizeof(T) == 8
#if !defined(UINT64_MAX)
      && sizeof(unsigned long long) == 8
#endif
     ));

  using type =
    std::conditional_t<sizeof(T) == 1,
#if defined(UINT8_MAX)
                       uint8_t,
#else
                       unsigned char,
#endif
    std::conditional_t<sizeof(T) == 2,
#if defined(UINT16_MAX)
                       uint16_t,
#else
                       unsigned short,
#endif
    std::conditional_t<sizeof(T) == 4,
#if defined(UINT32_MAX)
                       uint32_t,
#else
                       unsigned int,
#endif
#if defined(UINT64_MAX)
                       uint64_t>>>;
#else
                       unsigned long long>>>;
#endif
};


/*!
 * Atomic load using intrinsic
 */
template <typename T,
          std::enable_if_t<builtin_useIntrinsic<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicLoad(T *acc)
{
  return __atomic_load_n(acc, __ATOMIC_RELAXED);
}


/*!
 * Atomic store using intrinsic
 */
template <typename T,
          std::enable_if_t<builtin_useIntrinsic<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE void builtin_atomicStore(T *acc, T value)
{
  __atomic_store_n(acc, value, __ATOMIC_RELAXED);
}


/*!
 * Atomic exchange using intrinsic
 */
template <typename T,
          std::enable_if_t<builtin_useIntrinsic<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicExchange(T *acc, T value)
{
  return __atomic_exchange_n(acc, value, __ATOMIC_RELAXED);
}


/*!
 * Atomic compare and swap using intrinsic
 */
template <typename T,
          std::enable_if_t<builtin_useIntrinsic<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicCAS(T *acc, T compare, T value)
{
  __atomic_compare_exchange_n(
      acc, &compare, value, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  return compare;
}


#endif  // RAJA_COMPILER_MSVC


/*!
 * Alias for determining the integral type of the same size as the given type
 */
template <typename T>
using builtin_useReinterpret_t = typename builtin_useReinterpret<T>::type;


/*!
 * Atomic load using reinterpret cast
 */
template <typename T,
          std::enable_if_t<builtin_useReinterpret<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicLoad(T *acc)
{
  return RAJA::util::reinterp_A_as_B<builtin_useReinterpret_t<T>, T>(
    builtin_atomicLoad(
      reinterpret_cast<builtin_useReinterpret_t<T>*>(acc)));
}


/*!
 * Atomic store using reinterpret cast
 */
template <typename T,
          std::enable_if_t<builtin_useReinterpret<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE void builtin_atomicStore(T *acc, T value)
{
  builtin_atomicStore(
    reinterpret_cast<builtin_useReinterpret_t<T>*>(acc),
    RAJA::util::reinterp_A_as_B<T, builtin_useReinterpret_t<T>>(value));
}


/*!
 * Atomic exchange using reinterpret cast
 */
template <typename T,
          std::enable_if_t<builtin_useReinterpret<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicExchange(T *acc, T value)
{
  return RAJA::util::reinterp_A_as_B<builtin_useReinterpret_t<T>, T>(
    builtin_atomicExchange(
      reinterpret_cast<builtin_useReinterpret_t<T>*>(acc),
      RAJA::util::reinterp_A_as_B<T, builtin_useReinterpret_t<T>>(value)));
}


/*!
 * Atomic compare and swap using reinterpret cast
 */
template <typename T,
          std::enable_if_t<builtin_useReinterpret<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicCAS(T *acc, T compare, T value)
{
  return RAJA::util::reinterp_A_as_B<builtin_useReinterpret_t<T>, T>(
    builtin_atomicCAS(
      reinterpret_cast<builtin_useReinterpret_t<T>*>(acc),
      RAJA::util::reinterp_A_as_B<T, builtin_useReinterpret_t<T>>(compare),
      RAJA::util::reinterp_A_as_B<T, builtin_useReinterpret_t<T>>(value)));
}


/*!
 * Equality comparison for compare and swap loop. Converts to the underlying
 * integral type to avoid cases where the values will never compare equal
 * (most notably, NaNs).
 */
template <typename T,
          std::enable_if_t<builtin_useIntrinsic<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE bool builtin_atomicCAS_equal(const T &a, const T &b)
{
  return a == b;
}


template <typename T,
          std::enable_if_t<builtin_useReinterpret<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE bool builtin_atomicCAS_equal(const T &a, const T &b)
{
  return RAJA::util::reinterp_A_as_B<T, builtin_useReinterpret_t<T>>(a) ==
         RAJA::util::reinterp_A_as_B<T, builtin_useReinterpret_t<T>>(b);
}


/*!
 * Generic impementation of any atomic 8, 16, 32, or 64 bit operator
 * that can be implemented using a builtin compare and swap primitive.
 * Returns the OLD value that was replaced by the result of this operation.
 */
template <typename T, typename Oper>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicCAS(T *acc, Oper &&oper)
{
  T old = builtin_atomicLoad(acc);
  T expected;

  do {
    expected = old;
    old = builtin_atomicCAS(acc, expected, oper(expected));
  } while (!builtin_atomicCAS_equal(old, expected));

  return old;
}


/*!
 * Generic impementation of any atomic 8, 16, 32, or 64 bit operator
 * that can be implemented using a builtin compare and swap primitive.
 * Short-circuits for improved efficiency.
 * Returns the OLD value that was replaced by the result of this operation.
 */
template <typename T, typename Oper, typename ShortCircuit>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicCAS(T *acc, Oper &&oper, ShortCircuit &&sc)
{
  T old = builtin_atomicLoad(acc);

  if (sc(old)) {
    return old;
  }

  T expected;

  do {
    expected = old;
    old = builtin_atomicCAS(acc, expected, oper(expected));
  } while (!builtin_atomicCAS_equal(old, expected) && !sc(old));

  return old;
}


/*!
 * Atomics implemented using intrinsics
 */
#if defined(RAJA_COMPILER_MSVC) || (defined(_WIN32) && defined(__INTEL_COMPILER))


/*!
 * Atomic addition
 */
RAJA_INLINE char builtin_atomicAdd(char *acc, char value)
{
  return _InterlockedExchangeAdd8(acc, value);
}

RAJA_INLINE short builtin_atomicAdd(short *acc, short value)
{
  return _InterlockedExchangeAdd16(acc, value);
}

RAJA_INLINE long builtin_atomicAdd(long *acc, long value)
{
  return _InterlockedExchangeAdd(acc, value);
}

RAJA_INLINE long long builtin_atomicAdd(long long *acc, long long value)
{
  return _InterlockedExchangeAdd64(acc, value);
}


/*!
 * Atomic subtraction
 */
RAJA_INLINE char builtin_atomicSub(char *acc, char value)
{
  return _InterlockedExchangeAdd8(acc, -value);
}

RAJA_INLINE short builtin_atomicSub(short *acc, short value)
{
  return _InterlockedExchangeAdd16(acc, -value);
}

RAJA_INLINE long builtin_atomicSub(long *acc, long value)
{
  return _InterlockedExchangeAdd(acc, -value);
}

RAJA_INLINE long long builtin_atomicSub(long long *acc, long long value)
{
  return _InterlockedExchangeAdd64(acc, -value);
}


/*!
 * Atomic and
 */
RAJA_INLINE char builtin_atomicAnd(char *acc, char value)
{
  return _InterlockedAnd8(acc, value);
}

RAJA_INLINE short builtin_atomicAnd(short *acc, short value)
{
  return _InterlockedAnd16(acc, value);
}

RAJA_INLINE long builtin_atomicAnd(long *acc, long value)
{
  return _InterlockedAnd(acc, value);
}

RAJA_INLINE long long builtin_atomicAnd(long long *acc, long long value)
{
  return _InterlockedAnd64(acc, value);
}


/*!
 * Atomic or
 */
RAJA_INLINE char builtin_atomicOr(char *acc, char value)
{
  return _InterlockedOr8(acc, value);
}

RAJA_INLINE short builtin_atomicOr(short *acc, short value)
{
  return _InterlockedOr16(acc, value);
}

RAJA_INLINE long builtin_atomicOr(long *acc, long value)
{
  return _InterlockedOr(acc, value);
}

RAJA_INLINE long long builtin_atomicOr(long long *acc, long long value)
{
  return _InterlockedOr64(acc, value);
}


/*!
 * Atomic xor
 */
RAJA_INLINE char builtin_atomicXor(char *acc, char value)
{
  return _InterlockedXor8(acc, value);
}

RAJA_INLINE short builtin_atomicXor(short *acc, short value)
{
  return _InterlockedXor16(acc, value);
}

RAJA_INLINE long builtin_atomicXor(long *acc, long value)
{
  return _InterlockedXor(acc, value);
}

RAJA_INLINE long long builtin_atomicXor(long long *acc, long long value)
{
  return _InterlockedXor64(acc, value);
}


#else  // RAJA_COMPILER_MSVC


/*!
 * Atomic addition
 */
template <typename T,
          std::enable_if_t<builtin_useIntrinsic<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicAdd(T *acc, T value)
{
  return __atomic_fetch_add(acc, value, __ATOMIC_RELAXED);
}


/*!
 * Atomic subtraction
 */
template <typename T,
          std::enable_if_t<builtin_useIntrinsic<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicSub(T *acc, T value)
{
  return __atomic_fetch_sub(acc, value, __ATOMIC_RELAXED);
}


/*!
 * Atomic and
 */
template <typename T,
          std::enable_if_t<builtin_useIntrinsic<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicAnd(T *acc, T value)
{
  return __atomic_fetch_and(acc, value, __ATOMIC_RELAXED);
}


/*!
 * Atomic or
 */
template <typename T,
          std::enable_if_t<builtin_useIntrinsic<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicOr(T *acc, T value)
{
  return __atomic_fetch_or(acc, value, __ATOMIC_RELAXED);
}


/*!
 * Atomic xor
 */
template <typename T,
          std::enable_if_t<builtin_useIntrinsic<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicXor(T *acc, T value)
{
  return __atomic_fetch_xor(acc, value, __ATOMIC_RELAXED);
}


#endif  // RAJA_COMPILER_MSVC


/*!
 * Atomics implemented using compare and swap loop
 */


/*!
 * Type trait for determining if the operator should be implemented
 * using a compare and swap loop
 */
#if defined(RAJA_COMPILER_MSVC) || (defined(_WIN32) && defined(__INTEL_COMPILER))

template <typename T>
struct builtin_useCAS {
  static constexpr bool value =
    !std::is_same<T, char>::value &&
    !std::is_same<T, short>::value &&
    !std::is_same<T, long>::value &&
    !std::is_same<T, long long>::value &&
    (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);
};
#else  // RAJA_COMPILER_MSVC

template <typename T>
struct builtin_useCAS {
  static constexpr bool value =
    !std::is_integral<T>::value && !std::is_enum<T>::value &&
    (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);
};

#endif  // RAJA_COMPILER_MSVC


/*!
 * Atomic addition
 */
template <typename T,
          std::enable_if_t<builtin_useCAS<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicAdd(T *acc, T value)
{
  return builtin_atomicCAS(acc, [value] (T old) {
    return old + value;
  });
}


/*!
 * Atomic subtraction
 */
template <typename T,
          std::enable_if_t<builtin_useCAS<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicSub(T *acc, T value)
{
  return builtin_atomicCAS(acc, [value] (T old) {
    return old - value;
  });
}


/*!
 * Atomic and
 */
template <typename T,
          std::enable_if_t<builtin_useCAS<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicAnd(T *acc, T value)
{
  return builtin_atomicCAS(acc, [value] (T old) {
    return old & value;
  });
}


/*!
 * Atomic or
 */
template <typename T,
          std::enable_if_t<builtin_useCAS<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicOr(T *acc, T value)
{
  return builtin_atomicCAS(acc, [value] (T old) {
    return old | value;
  });
}


/*!
 * Atomic xor
 */
template <typename T,
          std::enable_if_t<builtin_useCAS<T>::value, bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicXor(T *acc, T value)
{
  return builtin_atomicCAS(acc, [value] (T old) {
    return old ^ value;
  });
}


}  // namespace detail


template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicLoad(builtin_atomic, T *acc)
{
  return detail::builtin_atomicLoad(acc);
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE void atomicStore(builtin_atomic, T *acc, T value)
{
  detail::builtin_atomicStore(acc, value);
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicAdd(builtin_atomic, T *acc, T value)
{
  return detail::builtin_atomicAdd(acc, value);
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicSub(builtin_atomic, T *acc, T value)
{
  return detail::builtin_atomicSub(acc, value);
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicMin(builtin_atomic, T *acc, T value)
{
  return detail::builtin_atomicCAS(acc,
                                   [value] (T old) {
                                     return value < old ? value : old;
                                   },
                                   [value] (T current) {
                                     return current <= value;
                                   });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicMax(builtin_atomic, T *acc, T value)
{
  return detail::builtin_atomicCAS(acc,
                                   [value] (T old) {
                                     return old < value ? value : old;
                                   },
                                   [value] (T current) {
                                     return value <= current;
                                   });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicInc(builtin_atomic, T *acc)
{
  return detail::builtin_atomicAdd(acc, static_cast<T>(1));
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicInc(builtin_atomic, T *acc, T value)
{
  return detail::builtin_atomicCAS(acc, [value] (T old) {
    return value <= old ? static_cast<T>(0) : old + static_cast<T>(1);
  });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicDec(builtin_atomic, T *acc)
{
  return detail::builtin_atomicSub(acc, static_cast<T>(1));
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicDec(builtin_atomic, T *acc, T value)
{
  return detail::builtin_atomicCAS(acc, [value] (T old) {
    return old == static_cast<T>(0) || value < old ? value : old - static_cast<T>(1);
  });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicAnd(builtin_atomic, T *acc, T value)
{
  return detail::builtin_atomicAnd(acc, value);
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicOr(builtin_atomic, T *acc, T value)
{
  return detail::builtin_atomicOr(acc, value);
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicXor(builtin_atomic, T *acc, T value)
{
  return detail::builtin_atomicXor(acc, value);
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicExchange(builtin_atomic, T *acc, T value)
{
  return detail::builtin_atomicExchange(acc, value);
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicCAS(builtin_atomic, T *acc, T compare, T value)
{
  return detail::builtin_atomicCAS(acc, compare, value);
}


}  // namespace RAJA


#endif
