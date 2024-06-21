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
 * Atomic load
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

template <typename T,
          std::enable_if_t<!std::is_same<T, char>::value &&
                           sizeof(T) == sizeof(char), bool> = true>
RAJA_INLINE T builtin_atomicLoad(T *acc)
{
  return RAJA::util::reinterp_A_as_B<char, T>(
    builtin_atomicLoad(reinterpret_cast<char*>(acc)));
}

template <typename T,
          std::enable_if_t<!std::is_same<T, short>::value &&
                           sizeof(T) == sizeof(short), bool> = true>
RAJA_INLINE T builtin_atomicLoad(T *acc)
{
  return RAJA::util::reinterp_A_as_B<short, T>(
    builtin_atomicLoad(reinterpret_cast<short*>(acc)));
}

template <typename T,
          std::enable_if_t<!std::is_same<T, long>::value &&
                           sizeof(T) == sizeof(long), bool> = true>
RAJA_INLINE T builtin_atomicLoad(T *acc)
{
  return RAJA::util::reinterp_A_as_B<long, T>(
    builtin_atomicLoad(reinterpret_cast<long*>(acc)));
}

template <typename T,
          std::enable_if_t<!std::is_same<T, long long>::value &&
                           sizeof(T) == sizeof(long long), bool> = true>
RAJA_INLINE T builtin_atomicLoad(T *acc)
{
  return RAJA::util::reinterp_A_as_B<long long, T>(
    builtin_atomicLoad(reinterpret_cast<long long*>(acc)));
}


/*!
 * Atomic exchange
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

template <typename T,
          std::enable_if_t<!std::is_same<T, char>::value &&
                           sizeof(T) == sizeof(char), bool> = true>
RAJA_INLINE T builtin_atomicExchange(T *acc, T value)
{
  return RAJA::util::reinterp_A_as_B<char, T>(
    builtin_atomicExchange(reinterpret_cast<char*>(acc),
                           RAJA::util::reinterp_A_as_B<T, char>(value)));
}

template <typename T,
          std::enable_if_t<!std::is_same<T, short>::value &&
                           sizeof(T) == sizeof(short), bool> = true>
RAJA_INLINE T builtin_atomicExchange(T *acc, T value)
{
  return RAJA::util::reinterp_A_as_B<short, T>(
    builtin_atomicExchange(reinterpret_cast<short*>(acc),
                           RAJA::util::reinterp_A_as_B<T, short>(value)));
}

template <typename T,
          std::enable_if_t<!std::is_same<T, long>::value &&
                           sizeof(T) == sizeof(long), bool> = true>
RAJA_INLINE T builtin_atomicExchange(T *acc, T value)
{
  return RAJA::util::reinterp_A_as_B<long, T>(
    builtin_atomicExchange(reinterpret_cast<long*>(acc),
                           RAJA::util::reinterp_A_as_B<T, long>(value)));
}

template <typename T,
          std::enable_if_t<!std::is_same<T, long long>::value &&
                           sizeof(T) == sizeof(long long), bool> = true>
RAJA_INLINE T builtin_atomicExchange(T *acc, T value)
{
  return RAJA::util::reinterp_A_as_B<long long, T>(
    builtin_atomicExchange(reinterpret_cast<long long*>(acc),
                           RAJA::util::reinterp_A_as_B<T, long long>(value)));
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
 * Atomic compare and swap
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

template <typename T,
          std::enable_if_t<!std::is_same<T, char>::value &&
                           sizeof(T) == sizeof(char), bool> = true>
RAJA_INLINE T builtin_atomicCAS(T *acc, T compare, T value)
{
  return RAJA::util::reinterp_A_as_B<char, T>(
    builtin_atomicCAS(reinterpret_cast<char*>(acc),
                      RAJA::util::reinterp_A_as_B<T, char>(value)));
}

template <typename T,
          std::enable_if_t<!std::is_same<T, short>::value &&
                           sizeof(T) == sizeof(short), bool> = true>
RAJA_INLINE T builtin_atomicCAS(T *acc, T compare, T value)
{
  return RAJA::util::reinterp_A_as_B<short, T>(
    builtin_atomicCAS(reinterpret_cast<short*>(acc),
                      RAJA::util::reinterp_A_as_B<T, short>(value)));
}

template <typename T,
          std::enable_if_t<!std::is_same<T, long>::value &&
                           sizeof(T) == sizeof(long), bool> = true>
RAJA_INLINE T builtin_atomicCAS(T *acc, T compare, T value)
{
  return RAJA::util::reinterp_A_as_B<long, T>(
    builtin_atomicCAS(reinterpret_cast<long*>(acc),
                      RAJA::util::reinterp_A_as_B<T, long>(value)));
}

template <typename T,
          std::enable_if_t<!std::is_same<T, long long>::value &&
                           sizeof(T) == sizeof(long long), bool> = true>
RAJA_INLINE T builtin_atomicCAS(T *acc, T compare, T value)
{
  return RAJA::util::reinterp_A_as_B<long long, T>(
    builtin_atomicCAS(reinterpret_cast<long long*>(acc),
                      RAJA::util::reinterp_A_as_B<T, long long>(value)));
}


/*!
 * Equality comparison for compare and swap loop. Converts to the underlying
 * integral type to avoid cases where the values will never compare equal
 * (most notably, NaNs).
 */
template <typename T,
          std::enable_if_t<std::is_same<T, char>::value ||
                           std::is_same<T, short>::value ||
                           std::is_same<T, long>::value ||
                           std::is_same<T, long long>::value, bool> = true>
RAJA_INLINE bool builtin_atomicCAS_equal(const T &a, const T &b)
{
  return a == b;
}

template <typename T,
          std::enable_if_t<!std::is_same<T, char>::value &&
                           sizeof(T) == sizeof(char), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE bool builtin_atomicCAS_equal(const T &a, const T &b)
{
  return RAJA::util::reinterp_A_as_B<T, char>(a) ==
         RAJA::util::reinterp_A_as_B<T, char>(b);
}

template <typename T,
          std::enable_if_t<!std::is_same<T, short>::value &&
                           sizeof(T) == sizeof(short), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE bool builtin_atomicCAS_equal(const T &a, const T &b)
{
  return RAJA::util::reinterp_A_as_B<T, short>(a) ==
         RAJA::util::reinterp_A_as_B<T, short>(b);
}

template <typename T,
          std::enable_if_t<!std::is_same<T, long>::value &&
                           sizeof(T) == sizeof(long), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE bool builtin_atomicCAS_equal(const T &a, const T &b)
{
  return RAJA::util::reinterp_A_as_B<T, long>(a) ==
         RAJA::util::reinterp_A_as_B<T, long>(b);
}

template <typename T,
          std::enable_if_t<!std::is_same<T, long long>::value &&
                           sizeof(T) == sizeof(long long), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE bool builtin_atomicCAS_equal(const T &a, const T &b)
{
  return RAJA::util::reinterp_A_as_B<T, long long>(a) ==
         RAJA::util::reinterp_A_as_B<T, long long>(b);
}


#else  // RAJA_COMPILER_MSVC


/*!
 * Atomic load, store, exchange, and compare and swap
 */
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
      acc, &compare, value, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
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
  return RAJA::util::reinterp_A_as_B<uint8_t, T>(
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

#else  // UINT8_MAX

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned char) &&
                           sizeof(unsigned char) == 1, bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicLoad(T *acc)
{
  return RAJA::util::reinterp_A_as_B<unsigned char, T>(
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

#endif  // UINT8_MAX

#if defined(UINT16_MAX)

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint16_t), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicLoad(T *acc)
{
  return RAJA::util::reinterp_A_as_B<uint16_t, T>(
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

#else  // UINT16_MAX

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned short) &&
                           sizeof(unsigned short) == 2, bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicLoad(T *acc)
{
  return RAJA::util::reinterp_A_as_B<unsigned short, T>(
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

#endif  // UINT16_MAX

#if defined(UINT32_MAX)

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint32_t), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicLoad(T *acc)
{
  return RAJA::util::reinterp_A_as_B<uint32_t, T>(
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

#else  // UINT32_MAX

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned int) &&
                           sizeof(unsigned int) == 4, bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicLoad(T *acc)
{
  return RAJA::util::reinterp_A_as_B<unsigned int, T>(
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

#endif  // UINT32_MAX

#if defined(UINT64_MAX)

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(uint64_t), bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicLoad(T *acc)
{
  return RAJA::util::reinterp_A_as_B<uint64_t, T>(
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

#else  // UINT64_MAX

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           sizeof(T) == sizeof(unsigned long long) &&
                           sizeof(unsigned long long) == 8, bool> = true>
RAJA_DEVICE_HIP
RAJA_INLINE T builtin_atomicLoad(T *acc)
{
  return RAJA::util::reinterp_A_as_B<unsigned long long, T>(
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

#endif  // UINT64_MAX


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

template <typename T,
          std::enable_if_t<!std::is_same<T, char>::value &&
                           !std::is_same<T, short>::value &&
                           !std::is_same<T, long>::value &&
                           !std::is_same<T, long long>::value &&
                           (sizeof(T) == 1 ||
                            sizeof(T) == 2 ||
                            sizeof(T) == 4 ||
                            sizeof(T) == 8), bool> = true>
RAJA_INLINE T builtin_atomicAdd(T *acc, T value)
{
  return builtin_atomicCAS(acc, [value] (T old) {
    return old + value;
  });
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

template <typename T,
          std::enable_if_t<!std::is_same<T, char>::value &&
                           !std::is_same<T, short>::value &&
                           !std::is_same<T, long>::value &&
                           !std::is_same<T, long long>::value &&
                           (sizeof(T) == 1 ||
                            sizeof(T) == 2 ||
                            sizeof(T) == 4 ||
                            sizeof(T) == 8), bool> = true>
RAJA_INLINE T builtin_atomicSub(T *acc, T value)
{
  return builtin_atomicCAS(acc, [value] (T old) {
    return old - value;
  });
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

template <typename T,
          std::enable_if_t<!std::is_same<T, char>::value &&
                           !std::is_same<T, short>::value &&
                           !std::is_same<T, long>::value &&
                           !std::is_same<T, long long>::value &&
                           (sizeof(T) == 1 ||
                            sizeof(T) == 2 ||
                            sizeof(T) == 4 ||
                            sizeof(T) == 8), bool> = true>
RAJA_INLINE T builtin_atomicAnd(T *acc, T value)
{
  return builtin_atomicCAS(acc, [value] (T old) {
    return old & value;
  });
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

template <typename T,
          std::enable_if_t<!std::is_same<T, char>::value &&
                           !std::is_same<T, short>::value &&
                           !std::is_same<T, long>::value &&
                           !std::is_same<T, long long>::value &&
                           (sizeof(T) == 1 ||
                            sizeof(T) == 2 ||
                            sizeof(T) == 4 ||
                            sizeof(T) == 8), bool> = true>
RAJA_INLINE T builtin_atomicOr(T *acc, T value)
{
  return builtin_atomicCAS(acc, [value] (T old) {
    return old | value;
  });
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

template <typename T,
          std::enable_if_t<!std::is_same<T, char>::value &&
                           !std::is_same<T, short>::value &&
                           !std::is_same<T, long>::value &&
                           !std::is_same<T, long long>::value &&
                           (sizeof(T) == 1 ||
                            sizeof(T) == 2 ||
                            sizeof(T) == 4 ||
                            sizeof(T) == 8), bool> = true>
RAJA_INLINE T builtin_atomicXor(T *acc, T value)
{
  return builtin_atomicCAS(acc, [value] (T old) {
    return old ^ value;
  });
}


#else  // RAJA_COMPILER_MSVC


/*
 * Atomic addition
 */
template <typename T,
          std::enable_if_t<(std::is_integral<T>::value ||
                            std::is_enum<T>::value) &&
                           (sizeof(T) == 1 ||
                            sizeof(T) == 2 ||
                            sizeof(T) == 4 ||
                            sizeof(T) == 8), bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicAdd(T *acc, T value)
{
  return __atomic_fetch_add(acc, value, __ATOMIC_RELAXED);
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           (sizeof(T) == 1 ||
                            sizeof(T) == 2 ||
                            sizeof(T) == 4 ||
                            sizeof(T) == 8), bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicAdd(T *acc, T value)
{
  return builtin_atomicCAS(acc, [value] (T old) {
    return old + value;
  });
}


/*
 * Atomic subtraction
 */
template <typename T,
          std::enable_if_t<(std::is_integral<T>::value ||
                            std::is_enum<T>::value) &&
                           (sizeof(T) == 1 ||
                            sizeof(T) == 2 ||
                            sizeof(T) == 4 ||
                            sizeof(T) == 8), bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicSub(T *acc, T value)
{
  return __atomic_fetch_sub(acc, value, __ATOMIC_RELAXED);
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           (sizeof(T) == 1 ||
                            sizeof(T) == 2 ||
                            sizeof(T) == 4 ||
                            sizeof(T) == 8), bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicSub(T *acc, T value)
{
  return builtin_atomicCAS(acc, [value] (T old) {
    return old - value;
  });
}


/*
 * Atomic and
 */
template <typename T,
          std::enable_if_t<(std::is_integral<T>::value ||
                            std::is_enum<T>::value) &&
                           (sizeof(T) == 1 ||
                            sizeof(T) == 2 ||
                            sizeof(T) == 4 ||
                            sizeof(T) == 8), bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicAnd(T *acc, T value)
{
  return __atomic_fetch_and(acc, value, __ATOMIC_RELAXED);
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           (sizeof(T) == 1 ||
                            sizeof(T) == 2 ||
                            sizeof(T) == 4 ||
                            sizeof(T) == 8), bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicAnd(T *acc, T value)
{
  return builtin_atomicCAS(acc, [value] (T old) {
    return old & value;
  });
}


/*
 * Atomic or
 */
template <typename T,
          std::enable_if_t<(std::is_integral<T>::value ||
                            std::is_enum<T>::value) &&
                           (sizeof(T) == 1 ||
                            sizeof(T) == 2 ||
                            sizeof(T) == 4 ||
                            sizeof(T) == 8), bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicOr(T *acc, T value)
{
  return __atomic_fetch_or(acc, value, __ATOMIC_RELAXED);
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           (sizeof(T) == 1 ||
                            sizeof(T) == 2 ||
                            sizeof(T) == 4 ||
                            sizeof(T) == 8), bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicOr(T *acc, T value)
{
  return builtin_atomicCAS(acc, [value] (T old) {
    return old | value;
  });
}


/*
 * Atomic xor
 */
template <typename T,
          std::enable_if_t<(std::is_integral<T>::value ||
                            std::is_enum<T>::value) &&
                           (sizeof(T) == 1 ||
                            sizeof(T) == 2 ||
                            sizeof(T) == 4 ||
                            sizeof(T) == 8), bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicXor(T *acc, T value)
{
  return __atomic_fetch_xor(acc, value, __ATOMIC_RELAXED);
}

template <typename T,
          std::enable_if_t<!std::is_integral<T>::value &&
                           !std::is_enum<T>::value &&
                           (sizeof(T) == 1 ||
                            sizeof(T) == 2 ||
                            sizeof(T) == 4 ||
                            sizeof(T) == 8), bool> = true>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomicXor(T *acc, T value)
{
  return builtin_atomicCAS(acc, [value] (T old) {
    return old ^ value;
  });
}


#endif  // RAJA_COMPILER_MSVC


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
