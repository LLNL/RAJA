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
  return RAJA::util::reinterp_A_as_B<unsigned int, T>(
    hip_atomicExchange(reinterpret_cast<unsigned int*>(acc),
                       RAJA::util::reinterp_A_as_B<T, unsigned int>(value)));
}

template <typename T,
          std::enable_if_t<!std::is_same<T, int>::value &&
                           !std::is_same<T, unsigned int>::value &&
                           !std::is_same<T, unsigned long long int>::value &&
                           !std::is_same<T, float>::value &&
                           sizeof(T) == sizeof(unsigned long long int), bool> = true>
RAJA_INLINE __device__ T hip_atomicExchange(T *acc, T value)
{
  return RAJA::util::reinterp_A_as_B<unsigned long long int, T>(
    hip_atomicExchange(reinterpret_cast<unsigned long long int*>(acc),
                       RAJA::util::reinterp_A_as_B<T, unsigned long long int>(value)));
}


/*!
 * Atomic load
 */
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
  return hip_atomicLoad(reinterpret_cast<uint8_t*>(acc));
}

#else

template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned char) &&
                           sizeof(unsigned char) == 1, bool> = true>
RAJA_INLINE __device__ T hip_atomicLoad(T *acc)
{
  return hip_atomicLoad(reinterpret_cast<unsigned char*>(acc));
}

#endif

#if defined(UINT16_MAX)

template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint16_t), bool> = true>
RAJA_INLINE __device__ T hip_atomicLoad(T *acc)
{
  return hip_atomicLoad(reinterpret_cast<uint16_t*>(acc));
}

#else

template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned short int) &&
                           sizeof(unsigned short int) == 2, bool> = true>
RAJA_INLINE __device__ T hip_atomicLoad(T *acc)
{
  return hip_atomicLoad(reinterpret_cast<unsigned short int*>(acc));
}

#endif

#if defined(UINT32_MAX)

template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint32_t), bool> = true>
RAJA_INLINE __device__ T hip_atomicLoad(T *acc)
{
  return hip_atomicLoad(reinterpret_cast<uint32_t*>(acc));
}

#else

template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned int) &&
                           sizeof(unsigned int) == 4, bool> = true>
RAJA_INLINE __device__ T hip_atomicLoad(T *acc)
{
  return hip_atomicLoad(reinterpret_cast<unsigned int*>(acc));
}

#endif

#if defined(UINT64_MAX)

template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint64_t), bool> = true>
RAJA_INLINE __device__ T hip_atomicLoad(T *acc)
{
  return hip_atomicLoad(reinterpret_cast<uint64_t*>(acc));
}

#else

template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned long long int) &&
                           sizeof(unsigned long long int) == 8, bool> = true>
RAJA_INLINE __device__ T hip_atomicLoad(T *acc)
{
  return hip_atomicLoad(reinterpret_cast<unsigned long long int*>(acc));
}

#endif

#else

template <typename T,
          std::enable_if_t<std::is_same<T, int>::value ||
                           std::is_same<T, unsigned int>::value ||
                           std::is_same<T, unsigned long long int>::value, bool> = true>
RAJA_INLINE __device__ T hip_atomicLoad(T *acc)
{
  return ::atomicOr(acc, static_cast<T>(0));
}

template <typename T,
          std::enable_if_t<!std::is_same<T, int>::value &&
                           !std::is_same<T, unsigned int>::value &&
                           !std::is_same<T, unsigned long long int>::value &&
                           sizeof(T) == sizeof(unsigned int), bool> = true>
RAJA_INLINE __device__ T hip_atomicLoad(T *acc)
{
  return RAJA::util::reinterp_A_as_B<unsigned int, T>(
    hip_atomicLoad(reinterpret_cast<unsigned int*>(acc)));
}

template <typename T,
          std::enable_if_t<!std::is_same<T, int>::value &&
                           !std::is_same<T, unsigned int>::value &&
                           !std::is_same<T, unsigned long long int>::value &&
                           sizeof(T) == sizeof(unsigned long long int), bool> = true>
RAJA_INLINE __device__ T hip_atomicLoad(T *acc)
{
   return RAJA::util::reinterp_A_as_B<unsigned long long int, T>(
     hip_atomicLoad(reinterpret_cast<unsigned long long int*>(acc)));
}

#endif


/*!
 * Atomic store
 */
#if defined(__has_builtin) && __has_builtin(__hip_atomic_store)

template <typename T,
          std::enable_if_t<std::is_arithmetic<T>::value ||
                           std::is_enum<T>::value, bool> = true>
RAJA_INLINE __device__ void hip_atomicStore(T *acc, T value)
{
  __hip_atomic_store(acc, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

#if defined(UINT8_MAX)

template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint8_t), bool> = true>
RAJA_INLINE __device__ void hip_atomicStore(T *acc, T value)
{
  hip_atomicStore(reinterpret_cast<uint8_t*>(acc),
                  RAJA::util::reinterp_A_as_B<T, uint8_t>(value));
}

#else

template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned char) &&
                           sizeof(unsigned char) == 1, bool> = true>
RAJA_INLINE __device__ void hip_atomicStore(T *acc, T value)
{
  hip_atomicStore(reinterpret_cast<unsigned char*>(acc),
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
  hip_atomicStore(reinterpret_cast<uint16_t*>(acc),
                  RAJA::util::reinterp_A_as_B<T, uint16_t>(value));
}

#else

template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned short int) &&
                           sizeof(unsigned short int) == 2, bool> = true>
RAJA_INLINE __device__ void hip_atomicStore(T *acc, T value)
{
  hip_atomicStore(reinterpret_cast<unsigned short int*>(acc),
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
  hip_atomicStore(reinterpret_cast<uint32_t*>(acc),
                  RAJA::util::reinterp_A_as_B<T, uint32_t>(value));
}

#else

template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned int) &&
                           sizeof(unsigned int) == 4, bool> = true>
RAJA_INLINE __device__ void hip_atomicStore(T *acc, T value)
{
  hip_atomicStore(reinterpret_cast<unsigned int*>(acc),
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
  hip_atomicStore(reinterpret_cast<uint64_t*>(acc),
                  RAJA::util::reinterp_A_as_B<T, uint64_t>(value));
}

#else

template <typename T,
          std::enable_if_t<!std::is_arithmetic<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned long long int) &&
                           sizeof(unsigned long long int) == 8, bool> = true>
RAJA_INLINE __device__ void hip_atomicStore(T *acc, T value)
{
  hip_atomicStore(reinterpret_cast<unsigned long long int*>(acc),
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
template <typename T,
          std::enable_if_t<std::is_same<T, int>::value ||
                           std::is_same<T, unsigned int>::value ||
                           std::is_same<T, unsigned long long int>::value, bool> = true>
RAJA_INLINE __device__ T hip_atomicCAS(T *acc, T compare, T value)
{
  return ::atomicCAS(acc, compare, value);
}

template <typename T,
          std::enable_if_t<!std::is_same<T, int>::value &&
                           !std::is_same<T, unsigned int>::value &&
                           !std::is_same<T, unsigned long long int>::value &&
                           sizeof(T) == sizeof(unsigned int), bool> = true>
RAJA_INLINE __device__ T hip_atomicCAS(T *acc, T compare, T value)
{
  return RAJA::util::reinterp_A_as_B<unsigned int, T>(
    hip_atomicCAS(reinterpret_cast<unsigned int*>(acc),
                  RAJA::util::reinterp_A_as_B<T, unsigned int>(compare),
                  RAJA::util::reinterp_A_as_B<T, unsigned int>(value)));
}

template <typename T,
          std::enable_if_t<!std::is_same<T, int>::value &&
                           !std::is_same<T, unsigned int>::value &&
                           !std::is_same<T, unsigned long long int>::value &&
                           sizeof(T) == sizeof(unsigned long long int), bool> = true>
RAJA_INLINE __device__ T hip_atomicCAS(T *acc, T compare, T value)
{
  return RAJA::util::reinterp_A_as_B<unsigned long long int, T>(
    hip_atomicCAS(reinterpret_cast<unsigned long long int*>(acc),
                  RAJA::util::reinterp_A_as_B<T, unsigned long long int>(compare),
                  RAJA::util::reinterp_A_as_B<T, unsigned long long int>(value)));
}


/*!
 * Equality comparison for compare and swap loop. Converts to the underlying
 * integral type to avoid cases where the values will never compare equal
 * (most notably, NaNs).
 */
template <typename T,
          std::enable_if_t<std::is_same<T, int>::value ||
                           std::is_same<T, unsigned int>::value ||
                           std::is_same<T, unsigned long long int>::value, bool> = true>
RAJA_INLINE __device__ bool hip_atomicCAS_equal(const T& a, const T& b)
{
  return a == b;
}

template <typename T,
          std::enable_if_t<!std::is_same<T, int>::value &&
                           !std::is_same<T, unsigned int>::value &&
                           !std::is_same<T, unsigned long long int>::value &&
                           sizeof(T) == sizeof(unsigned int), bool> = true>
RAJA_INLINE __device__ bool hip_atomicCAS_equal(const T& a, const T& b)
{
  return RAJA::util::reinterp_A_as_B<T, unsigned int>(a) ==
         RAJA::util::reinterp_A_as_B<T, unsigned int>(b);
}

template <typename T,
          std::enable_if_t<!std::is_same<T, int>::value &&
                           !std::is_same<T, unsigned int>::value &&
                           !std::is_same<T, unsigned long long int>::value &&
                           sizeof(T) == sizeof(unsigned long long int), bool> = true>
RAJA_INLINE __device__ bool hip_atomicCAS_equal(const T& a, const T& b)
{
  return RAJA::util::reinterp_A_as_B<T, unsigned long long int>(a) ==
         RAJA::util::reinterp_A_as_B<T, unsigned long long int>(b);
}


/*!
 * Generic impementation of any atomic 32-bit or 64-bit operator.
 * Implementation uses the existing HIP supplied unsigned 32-bit or 64-bit CAS
 * operator. Returns the OLD value that was replaced by the result of this
 * operation.
 */
template <typename T, typename Oper>
RAJA_INLINE __device__ T hip_atomicCAS(T *acc, Oper&& oper)
{
  T old = hip_atomicLoad(acc);
  T expected;

  do {
    expected = old;
    old = hip_atomicCAS(acc, expected, oper(expected));
  } while (!hip_atomicCAS_equal(old, expected));

  return old;
}


/*!
 * Generic impementation of any atomic 32-bit or 64-bit operator with short-circuiting.
 * Implementation uses the existing HIP supplied unsigned 32-bit or 64-bit CAS
 * operator. Returns the OLD value that was replaced by the result of this
 * operation.
 */
template <typename T, typename Oper, typename ShortCircuit>
RAJA_INLINE __device__ T hip_atomicCAS(T *acc, Oper&& oper, ShortCircuit&& sc)
{
  T old = hip_atomicLoad(acc);

  if (sc(old)) {
    return old;
  }

  T expected;

  do {
    expected = old;
    old = hip_atomicCAS(acc, expected, oper(expected));
  } while (!hip_atomicCAS_equal(old, expected) && !sc(old));

  return old;
}


/*!
 * Atomic addition
 */
using hip_atomicAdd_builtin_types = list<
      int
     ,unsigned int
     ,unsigned long long
     ,float
#ifdef RAJA_ENABLE_HIP_DOUBLE_ATOMICADD
     ,double
#endif
    >;

template <typename T,
          enable_if_is_none_of<T, hip_atomicAdd_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicAdd(T *acc, T value)
{
  return hip_atomicCAS(acc, [value] (T old) {
    return old + value;
  });
}

template <typename T,
          enable_if_is_any_of<T, hip_atomicAdd_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicAdd(T *acc, T value)
{
  return ::atomicAdd(acc, value);
}


/*!
 * Atomic subtraction
 */

/*!
 * List of types where HIP builtin atomics are used to implement atomicSub.
 */
using hip_atomicSub_builtin_types = list<
      int
     ,unsigned int
     ,unsigned long long
     ,float
#ifdef RAJA_ENABLE_HIP_DOUBLE_ATOMICADD
     ,double
#endif
    >;

/*!
 * List of types where HIP builtin atomicSub is used to implement atomicSub.
 *
 * Avoid multiple definition errors by including the previous list type here
 * to ensure these lists have different types.
 */
using hip_atomicSub_via_Sub_builtin_types = list<
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

template <typename T,
          enable_if_is_none_of<T, hip_atomicSub_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicSub(T *acc, T value)
{
  return hip_atomicCAS(acc, [value] (T old) {
    return old - value;
  });
}

/*!
 * HIP atomicSub builtin implementation.
 */
template <typename T,
          enable_if_is_any_of<T, hip_atomicSub_via_Sub_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicSub(T *acc, T value)
{
  return ::atomicSub(acc, value);
}

/*!
 * HIP atomicSub via atomicAdd builtin implementation.
 */
template <typename T,
          enable_if_is_any_of<T, hip_atomicSub_via_Add_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicSub(T *acc, T value)
{
  return ::atomicAdd(acc, -value);
}


/*!
 * Atomic minimum
 */
using hip_atomicMin_builtin_types = hip_atomicCommon_builtin_types;

template <typename T,
          enable_if_is_none_of<T, hip_atomicMin_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicMin(T *acc, T value)
{
  return hip_atomicCAS(acc,
                       [value] (T old) {
                         return value < old ? value : old;
                       },
                       [value] (T current) {
                         return current < value;
                       });
}

template <typename T,
          enable_if_is_any_of<T, hip_atomicMin_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicMin(T *acc, T value)
{
  return ::atomicMin(acc, value);
}


/*!
 * Atomic maximum
 */
using hip_atomicMax_builtin_types = hip_atomicCommon_builtin_types;

template <typename T,
          enable_if_is_none_of<T, hip_atomicMax_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicMax(T *acc, T value)
{
  return hip_atomicCAS(acc,
                       [value] (T old) {
                         return old < value ? value : old;
                       },
                       [value] (T current) {
                         return value < current;
                       });
}

template <typename T,
          enable_if_is_any_of<T, hip_atomicMax_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicMax(T *acc, T value)
{
  return ::atomicMax(acc, value);
}


/*!
 * Atomic increment with reset
 */
template <typename T>
RAJA_INLINE __device__ T hip_atomicInc(T *acc, T value)
{
  return hip_atomicCAS(acc, [value] (T old) {
    return value <= old ? static_cast<T>(0) : old + static_cast<T>(1);
  });
}


/*!
 * Atomic increment (implemented in terms of atomic addition)
 */
template <typename T>
RAJA_INLINE __device__ T hip_atomicInc(T *acc)
{
  return hip_atomicAdd(acc, static_cast<T>(1));
}


/*!
 * Atomic decrement with reset
 */
template <typename T>
RAJA_INLINE __device__ T hip_atomicDec(T *acc, T value)
{
  return hip_atomicCAS(acc, [value] (T old) {
    return old == static_cast<T>(0) || value < old ? value : old - static_cast<T>(1);
  });
}


/*!
 * Atomic decrement (implemented in terms of atomic subtraction)
 */
template <typename T>
RAJA_INLINE __device__ T hip_atomicDec(T *acc)
{
  return hip_atomicSub(acc, static_cast<T>(1));
}


/*!
 * Atomic and
 */
using hip_atomicAnd_builtin_types = hip_atomicCommon_builtin_types;

template <typename T,
          enable_if_is_none_of<T, hip_atomicAnd_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicAnd(T *acc, T value)
{
  return hip_atomicCAS(acc, [value] (T old) {
    return old & value;
  });
}

template <typename T,
          enable_if_is_any_of<T, hip_atomicAnd_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicAnd(T *acc, T value)
{
  return ::atomicAnd(acc, value);
}


/*!
 * Atomic or
 */
using hip_atomicOr_builtin_types = hip_atomicCommon_builtin_types;

template <typename T,
          enable_if_is_none_of<T, hip_atomicOr_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicOr(T *acc, T value)
{
  return hip_atomicCAS(acc, [value] (T old) {
    return old | value;
  });
}

template <typename T,
          enable_if_is_any_of<T, hip_atomicOr_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicOr(T *acc, T value)
{
  return ::atomicOr(acc, value);
}


/*!
 * Atomic xor
 */
using hip_atomicXor_builtin_types = hip_atomicCommon_builtin_types;

template <typename T,
          enable_if_is_none_of<T, hip_atomicXor_builtin_types>* = nullptr>
RAJA_INLINE __device__ T hip_atomicXor(T *acc, T value)
{
  return hip_atomicCAS(acc, [value] (T old) {
    return old ^ value;
  });
}

template <typename T,
          enable_if_is_any_of<T, hip_atomicXor_builtin_types>* = nullptr>
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
