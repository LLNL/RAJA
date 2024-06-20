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

#include "RAJA/util/TypeConvert.hpp"
#include "RAJA/util/macros.hpp"

#include <cstdint>

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

namespace detail
{

template <std::size_t BYTES>
struct BuiltinAtomicTypeImpl {
  static_assert(!(BYTES == sizeof(unsigned) ||
                  BYTES == sizeof(unsigned long long)),
                "Builtin atomic operations require targets that match the size of 'unsigned int' or 'unsigned long long' (usually 4 or 8 bytes).");
};

template <>
struct BuiltinAtomicTypeImpl<sizeof(unsigned)> {
  using type = unsigned;
};

template <>
struct BuiltinAtomicTypeImpl<sizeof(unsigned long long)> {
  using type = unsigned long long;
};

template <class T>
using BuiltinAtomicType = typename BuiltinAtomicTypeImpl<sizeof(T)>::type;

#if defined(RAJA_COMPILER_MSVC) || (defined(_WIN32) && defined(__INTEL_COMPILER))

RAJA_INLINE unsigned builtin_atomicLoad(unsigned *acc)
{
  static_assert(sizeof(unsigned) == sizeof(long),
                "builtin atomic load assumes unsigned and long are the same size");

  return RAJA::util::reinterp_A_as_B<long, unsigned>(_InterlockedOr((long *)acc, 0));
}

RAJA_INLINE unsigned long long builtin_atomicLoad(
    unsigned long long *acc)
{
  static_assert(sizeof(unsigned long long) == sizeof(long long),
                "builtin atomic load assumes unsigned long long and long long are the same size");

  return RAJA::util::reinterp_A_as_B<long long, unsigned long long>(_InterlockedOr64((long long *)acc, 0));
}

RAJA_INLINE void builtin_atomicStore(unsigned *acc, unsigned value)
{
  static_assert(sizeof(unsigned) == sizeof(long),
                "builtin atomic store assumes unsigned and long are the same size");

  _InterlockedExchange((long *)acc, RAJA::util::reinterp_A_as_B<unsigned, long>(value));
}

RAJA_INLINE void builtin_atomicStore(
    unsigned long long *acc,
    unsigned long long value)
{
  static_assert(sizeof(unsigned long long) == sizeof(long long),
                "builtin atomic store assumes unsigned long long and long long are the same size");

  _InterlockedExchange64((long long *)acc, RAJA::util::reinterp_A_as_B<unsigned long long, long long>(value));
}

RAJA_INLINE unsigned builtin_atomicCAS(unsigned *acc,
                                        unsigned compare,
                                        unsigned value)
{

  long long_value = RAJA::util::reinterp_A_as_B<unsigned, long>(value);
  long long_compare = RAJA::util::reinterp_A_as_B<unsigned, long>(compare);

  long old = _InterlockedCompareExchange((long *)acc, long_value, long_compare);

  return RAJA::util::reinterp_A_as_B<long, unsigned>(old);
}

RAJA_INLINE unsigned long long builtin_atomicCAS(
    unsigned long long *acc,
    unsigned long long compare,
    unsigned long long value)
{

  long long long_value =
      RAJA::util::reinterp_A_as_B<unsigned long long, long long>(value);
  long long long_compare =
      RAJA::util::reinterp_A_as_B<unsigned long long, long long>(compare);

  long long old = _InterlockedCompareExchange64((long long *)acc,
                                                long_value,
                                                long_compare);

  return RAJA::util::reinterp_A_as_B<long long, unsigned long long>(old);
}

#else  // RAJA_COMPILER_MSVC

template <typename T,
          std::enable_if_t<(std::is_integral<T>::value ||
                            std::is_enum<T>::value) &&
                           (sizeof(T) == 1 ||
                            sizeof(T) == 2 ||
                            sizeof(T) == 4 ||
                            sizeof(T) == 8), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicLoad(T *acc)
{
  return __atomic_load_n(acc, __ATOMIC_RELAXED);
}

template <typename T,
          std::enable_if_t<(std::is_integral<T>::value ||
                            std::is_enum<T>::value) &&
                           (sizeof(T) == 1 ||
                            sizeof(T) == 2 ||
                            sizeof(T) == 4 ||
                            sizeof(T) == 8), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE void builtin_atomicStore(T *acc, T value)
{
  __atomic_store_n(acc, value, __ATOMIC_RELAXED);
}

template <typename T,
          std::enable_if_t<(std::is_integral<T>::value ||
                            std::is_enum<T>::value) &&
                           (sizeof(T) == 1 ||
                            sizeof(T) == 2 ||
                            sizeof(T) == 4 ||
                            sizeof(T) == 8), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicExchange(T *acc, T value)
{
  return __atomic_exchange_n(acc, value, __ATOMIC_RELAXED);
}

template <typename T,
          std::enable_if_t<(std::is_integral<T>::value ||
                            std::is_enum<T>::value) &&
                           (sizeof(T) == 1 ||
                            sizeof(T) == 2 ||
                            sizeof(T) == 4 ||
                            sizeof(T) == 8), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicCAS(T *acc, T compare, T value)
{
  __atomic_compare_exchange_n(
      acc, &compare, value, false, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED);
  return compare;
}

template <typename T,
          std::enable_if_t<(std::is_integral<T>::value ||
                            std::is_enum<T>::value) &&
                           (sizeof(T) == 1 ||
                            sizeof(T) == 2 ||
                            sizeof(T) == 4 ||
                            sizeof(T) == 8), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE bool builtin_atomicCAS_equal(const T &a, const T &b)
{
  return a == b;
}

#if defined(UINT8_MAX)

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint8_t), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicLoad(T *acc)
{
  return RAJA::util_reinterp_A_as_B<uint8_t, T>(
    builtin_atomicLoad(reinterpret_cast<uint8_t*>(acc)));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint8_t), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE void builtin_atomicStore(T *acc, T value)
{
  builtin_atomicStore(reinterpret_cast<uint8_t*>(acc),
                      RAJA::util::reinterp_A_as_B<T, uint8_t>(value));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint8_t), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicExchange(T *acc, T value)
{
  return RAJA::util::reinterp_A_as_B<uint8_t, T>(
    builtin_atomicExchange(reinterpret_cast<uint8_t*>(acc),
                           RAJA::util::reinterp_A_as_B<T, uint8_t>(value)));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint8_t), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicCAS(T *acc, T compare, T value)
{
  return RAJA::util::reinterp_A_as_B<uint8_t, T>(
    builtin_atomicCAS(reinterpret_cast<uint8_t*>(acc),
                      RAJA::util::reinterp_A_as_B<T, uint8_t>(compare),
                      RAJA::util::reinterp_A_as_B<T, uint8_t>(value)));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint8_t), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE bool builtin_atomicCAS_equal(const T &a, const T &b)
{
  return RAJA::util::reinterp_A_as_B<T, uint8_t>(a) ==
         RAJA::util::reinterp_A_as_B<T, uint8_t>(b);
}

#else

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned char) &&
                           sizeof(unsigned char) == 1, bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicLoad(T *acc)
{
  return RAJA::util_reinterp_A_as_B<unsigned char, T>(
    builtin_atomicLoad(reinterpret_cast<unsigned char*>(acc)));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned char) &&
                           sizeof(unsigned char) == 1, bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE void builtin_atomicStore(T *acc, T value)
{
  builtin_atomicStore(reinterpret_cast<unsigned char*>(acc),
                      RAJA::util::reinterp_A_as_B<T, unsigned char>(value));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned char) &&
                           sizeof(unsigned char) == 1, bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicExchange(T *acc, T value)
{
  return RAJA::util::reinterp_A_as_B<unsigned char, T>(
    builtin_atomicExchange(reinterpret_cast<unsigned char*>(acc),
                           RAJA::util::reinterp_A_as_B<T, unsigned char>(value)));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned char) &&
                           sizeof(unsigned char) == 1, bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicCAS(T *acc, T compare, T value)
{
  return RAJA::util::reinterp_A_as_B<unsigned char, T>(
    builtin_atomicCAS(reinterpret_cast<unsigned char*>(acc),
                      RAJA::util::reinterp_A_as_B<T, unsigned char>(compare),
                      RAJA::util::reinterp_A_as_B<T, unsigned char>(value)));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned char) &&
                           sizeof(unsigned char) == 1, bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE bool builtin_atomicCAS_equal(const T &a, const T &b)
{
  return RAJA::util::reinterp_A_as_B<T, unsigned char>(a) ==
         RAJA::util::reinterp_A_as_B<T, unsigned char>(b);
}

#endif

#if defined(UINT16_MAX)

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint16_t), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicLoad(T *acc)
{
  return RAJA::util_reinterp_A_as_B<uint16_t, T>(
    builtin_atomicLoad(reinterpret_cast<uint16_t*>(acc)));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint16_t), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE void builtin_atomicStore(T *acc, T value)
{
  builtin_atomicStore(reinterpret_cast<uint16_t*>(acc),
                      RAJA::util::reinterp_A_as_B<T, uint16_t>(value));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint16_t), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicExchange(T *acc, T value)
{
  return RAJA::util::reinterp_A_as_B<uint16_t, T>(
    builtin_atomicExchange(reinterpret_cast<uint16_t*>(acc),
                           RAJA::util::reinterp_A_as_B<T, uint16_t>(value)));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint16_t), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicCAS(T *acc, T compare, T value)
{
  return RAJA::util::reinterp_A_as_B<uint16_t, T>(
    builtin_atomicCAS(reinterpret_cast<uint16_t*>(acc),
                      RAJA::util::reinterp_A_as_B<T, uint16_t>(compare),
                      RAJA::util::reinterp_A_as_B<T, uint16_t>(value)));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint16_t), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE bool builtin_atomicCAS_equal(const T &a, const T &b)
{
  return RAJA::util::reinterp_A_as_B<T, uint16_t>(a) ==
         RAJA::util::reinterp_A_as_B<T, uint16_t>(b);
}

#else

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned short) &&
                           sizeof(unsigned short) == 2, bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicLoad(T *acc)
{
  return RAJA::util_reinterp_A_as_B<unsigned short, T>(
    builtin_atomicLoad(reinterpret_cast<unsigned short*>(acc)));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned short) &&
                           sizeof(unsigned short) == 2, bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE void builtin_atomicStore(T *acc, T value)
{
  builtin_atomicStore(reinterpret_cast<unsigned short*>(acc),
                      RAJA::util::reinterp_A_as_B<T, unsigned short>(value));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned short) &&
                           sizeof(unsigned short) == 2, bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicExchange(T *acc, T value)
{
  return RAJA::util::reinterp_A_as_B<unsigned short, T>(
    builtin_atomicExchange(reinterpret_cast<unsigned short*>(acc),
                           RAJA::util::reinterp_A_as_B<T, unsigned short>(value)));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned short) &&
                           sizeof(unsigned short) == 2, bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicCAS(T *acc, T compare, T value)
{
  return RAJA::util::reinterp_A_as_B<unsigned short, T>(
    builtin_atomicCAS(reinterpret_cast<unsigned short*>(acc),
                      RAJA::util::reinterp_A_as_B<T, unsigned short>(compare),
                      RAJA::util::reinterp_A_as_B<T, unsigned short>(value)));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned short) &&
                           sizeof(unsigned short) == 2, bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE bool builtin_atomicCAS_equal(const T &a, const T &b)
{
  return RAJA::util::reinterp_A_as_B<T, unsigned short>(a) ==
         RAJA::util::reinterp_A_as_B<T, unsigned short>(b);
}

#endif

#if defined(UINT32_MAX)

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint32_t), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicLoad(T *acc)
{
  return RAJA::util_reinterp_A_as_B<uint32_t, T>(
    builtin_atomicLoad(reinterpret_cast<uint32_t*>(acc)));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint32_t), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE void builtin_atomicStore(T *acc, T value)
{
  builtin_atomicStore(reinterpret_cast<uint32_t*>(acc),
                      RAJA::util::reinterp_A_as_B<T, uint32_t>(value));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint32_t), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicExchange(T *acc, T value)
{
  return RAJA::util::reinterp_A_as_B<uint32_t, T>(
    builtin_atomicExchange(reinterpret_cast<uint32_t*>(acc),
                           RAJA::util::reinterp_A_as_B<T, uint32_t>(value)));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint32_t), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicCAS(T *acc, T compare, T value)
{
  return RAJA::util::reinterp_A_as_B<uint32_t, T>(
    builtin_atomicCAS(reinterpret_cast<uint32_t*>(acc),
                      RAJA::util::reinterp_A_as_B<T, uint32_t>(compare),
                      RAJA::util::reinterp_A_as_B<T, uint32_t>(value)));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint32_t), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE bool builtin_atomicCAS_equal(const T &a, const T &b)
{
  return RAJA::util::reinterp_A_as_B<T, uint32_t>(a) ==
         RAJA::util::reinterp_A_as_B<T, uint32_t>(b);
}

#else

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned int) &&
                           sizeof(unsigned int) == 4, bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicLoad(T *acc)
{
  return RAJA::util_reinterp_A_as_B<unsigned int, T>(
    builtin_atomicLoad(reinterpret_cast<unsigned int*>(acc)));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned int) &&
                           sizeof(unsigned int) == 4, bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE void builtin_atomicStore(T *acc, T value)
{
  builtin_atomicStore(reinterpret_cast<unsigned int*>(acc),
                      RAJA::util::reinterp_A_as_B<T, unsigned int>(value));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned int) &&
                           sizeof(unsigned int) == 4, bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicExchange(T *acc, T value)
{
  return RAJA::util::reinterp_A_as_B<unsigned int, T>(
    builtin_atomicExchange(reinterpret_cast<unsigned int*>(acc),
                           RAJA::util::reinterp_A_as_B<T, unsigned int>(value)));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned int) &&
                           sizeof(unsigned int) == 4, bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicCAS(T *acc, T compare, T value)
{
  return RAJA::util::reinterp_A_as_B<unsigned int, T>(
    builtin_atomicCAS(reinterpret_cast<unsigned int*>(acc),
                      RAJA::util::reinterp_A_as_B<T, unsigned int>(compare),
                      RAJA::util::reinterp_A_as_B<T, unsigned int>(value)));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned int) &&
                           sizeof(unsigned int) == 4, bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE bool builtin_atomicCAS_equal(const T &a, const T &b)
{
  return RAJA::util::reinterp_A_as_B<T, unsigned int>(a) ==
         RAJA::util::reinterp_A_as_B<T, unsigned int>(b);
}

#endif

#if defined(UINT64_MAX)

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint64_t), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicLoad(T *acc)
{
  return RAJA::util_reinterp_A_as_B<uint64_t, T>(
    builtin_atomicLoad(reinterpret_cast<uint64_t*>(acc)));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint64_t), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE void builtin_atomicStore(T *acc, T value)
{
  builtin_atomicStore(reinterpret_cast<uint64_t*>(acc),
                      RAJA::util::reinterp_A_as_B<T, uint64_t>(value));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint64_t), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicExchange(T *acc, T value)
{
  return RAJA::util::reinterp_A_as_B<uint64_t, T>(
    builtin_atomicExchange(reinterpret_cast<uint64_t*>(acc),
                           RAJA::util::reinterp_A_as_B<T, uint64_t>(value)));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint64_t), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicCAS(T *acc, T compare, T value)
{
  return RAJA::util::reinterp_A_as_B<uint64_t, T>(
    builtin_atomicCAS(reinterpret_cast<uint64_t*>(acc),
                      RAJA::util::reinterp_A_as_B<T, uint64_t>(compare),
                      RAJA::util::reinterp_A_as_B<T, uint64_t>(value)));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint64_t), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE bool builtin_atomicCAS_equal(const T &a, const T &b)
{
  return RAJA::util::reinterp_A_as_B<T, uint64_t>(a) ==
         RAJA::util::reinterp_A_as_B<T, uint64_t>(b);
}

#else

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned long long) &&
                           sizeof(unsigned long long) == 8, bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicLoad(T *acc)
{
  return RAJA::util_reinterp_A_as_B<unsigned long long, T>(
    builtin_atomicLoad(reinterpret_cast<unsigned long long*>(acc)));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned long long) &&
                           sizeof(unsigned long long) == 8, bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE void builtin_atomicStore(T *acc, T value)
{
  builtin_atomicStore(reinterpret_cast<unsigned long long*>(acc),
                      RAJA::util::reinterp_A_as_B<T, unsigned long long>(value));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned long long) &&
                           sizeof(unsigned long long) == 4, bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicExchange(T *acc, T value)
{
  return RAJA::util::reinterp_A_as_B<unsigned long long, T>(
    builtin_atomicExchange(reinterpret_cast<unsigned long long*>(acc),
                           RAJA::util::reinterp_A_as_B<T, unsigned long long>(value)));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned long long) &&
                           sizeof(unsigned long long) == 8, bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicCAS(T *acc, T compare, T value)
{
  return RAJA::util::reinterp_A_as_B<unsigned long long, T>(
    builtin_atomicCAS(reinterpret_cast<unsigned long long*>(acc),
                      RAJA::util::reinterp_A_as_B<T, unsigned long long>(compare),
                      RAJA::util::reinterp_A_as_B<T, unsigned long long>(value)));
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned long long) &&
                           sizeof(unsigned long long) == 4, bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE bool builtin_atomicCAS_equal(const T &a, const T &b)
{
  return RAJA::util::reinterp_A_as_B<T, unsigned long long>(a) ==
         RAJA::util::reinterp_A_as_B<T, unsigned long long>(b);
}

#endif

#endif  // RAJA_COMPILER_MSVC


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
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicCAS(T *acc,
                                                Oper &&oper,
                                                ShortCircuit &&sc)
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


}  // namespace detail


template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicLoad(builtin_atomic,
                                         T *acc)
{
  return detail::builtin_atomicLoad(acc);
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE void atomicStore(builtin_atomic,
                                             T *acc,
                                             T value)
{
  detail::builtin_atomicStore(acc, value);
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicAdd(builtin_atomic,
                                        T *acc,
                                        T value)
{
  return detail::builtin_atomicCAS(acc, [=](T a) { return a + value; });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicSub(builtin_atomic,
                                        T *acc,
                                        T value)
{
  return detail::builtin_atomicCAS(acc, [=](T a) { return a - value; });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicMin(builtin_atomic,
                                        T *acc,
                                        T value)
{
  return detail::builtin_atomicCAS(acc,
                                            [=](T a) {
                                              return a < value ? a : value;
                                            },
                                            [=](T current) {
                                              return current <= value;
                                            });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicMax(builtin_atomic,
                                        T *acc,
                                        T value)
{
  return detail::builtin_atomicCAS(acc,
                                            [=](T a) {
                                              return a > value ? a : value;
                                            },
                                            [=](T current) {
                                              return value <= current;
                                            });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicInc(builtin_atomic, T *acc)
{
  return detail::builtin_atomicCAS(acc, [=](T a) { return a + 1; });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicInc(builtin_atomic, T *acc, T val)
{
  return detail::builtin_atomicCAS(acc, [=](T old) {
    return ((old >= val) ? 0 : (old + 1));
  });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicDec(builtin_atomic, T *acc)
{
  return detail::builtin_atomicCAS(acc, [=](T a) { return a - 1; });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicDec(builtin_atomic, T *acc, T val)
{
  return detail::builtin_atomicCAS(acc, [=](T old) {
    return (((old == 0) | (old > val)) ? val : (old - 1));
  });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicAnd(builtin_atomic,
                                        T *acc,
                                        T value)
{
  return detail::builtin_atomicCAS(acc, [=](T a) { return a & value; });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicOr(builtin_atomic, T *acc, T value)
{
  return detail::builtin_atomicCAS(acc, [=](T a) { return a | value; });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicXor(builtin_atomic,
                                        T *acc,
                                        T value)
{
  return detail::builtin_atomicCAS(acc, [=](T a) { return a ^ value; });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicExchange(builtin_atomic,
                                             T *acc,
                                             T value)
{
  return detail::builtin_atomicCAS(acc, [=](T) { return value; });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T
atomicCAS(builtin_atomic, T *acc, T compare, T value)
{
  return detail::builtin_atomicCAS(acc, compare, value);
}


}  // namespace RAJA

// make sure this define doesn't bleed out of this header
#undef RAJA_AUTO_ATOMIC

#endif
