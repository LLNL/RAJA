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

RAJA_INLINE unsigned builtin_atomic_load(unsigned *acc)
{
  static_assert(sizeof(unsigned) == sizeof(long),
                "builtin atomic load assumes unsigned and long are the same size");

  return RAJA::util::reinterp_A_as_B<long, unsigned>(_InterlockedOr((long *)acc, 0));
}

RAJA_INLINE unsigned long long builtin_atomic_load(
    unsigned long long *acc)
{
  static_assert(sizeof(unsigned long long) == sizeof(long long),
                "builtin atomic load assumes unsigned long long and long long are the same size");

  return RAJA::util::reinterp_A_as_B<long long, unsigned long long>(_InterlockedOr64((long long *)acc, 0));
}

RAJA_INLINE void builtin_atomic_store(unsigned *acc, unsigned value)
{
  static_assert(sizeof(unsigned) == sizeof(long),
                "builtin atomic store assumes unsigned and long are the same size");

  _InterlockedExchange((long *)acc, RAJA::util::reinterp_A_as_B<unsigned, long>(value));
}

RAJA_INLINE void builtin_atomic_store(
    unsigned long long *acc,
    unsigned long long value)
{
  static_assert(sizeof(unsigned long long) == sizeof(long long),
                "builtin atomic store assumes unsigned long long and long long are the same size");

  _InterlockedExchange64((long long *)acc, RAJA::util::reinterp_A_as_B<unsigned long long, long long>(value));
}

RAJA_INLINE unsigned builtin_atomic_CAS(unsigned *acc,
                                        unsigned compare,
                                        unsigned value)
{

  long long_value = RAJA::util::reinterp_A_as_B<unsigned, long>(value);
  long long_compare = RAJA::util::reinterp_A_as_B<unsigned, long>(compare);

  long old = _InterlockedCompareExchange((long *)acc, long_value, long_compare);

  return RAJA::util::reinterp_A_as_B<long, unsigned>(old);
}

RAJA_INLINE unsigned long long builtin_atomic_CAS(
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
                      reinterpret_cast<uint8_t&>(value));
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
                           reinterpret_cast<uint8_t&>(value)));
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
                      reinterpret_cast<uint8_t&>(compare),
                      reinterpret_cast<uint8_t&>(value)));
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
                      reinterpret_cast<unsigned char&>(value));
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
                           reinterpret_cast<unsigned char&>(value)));
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
                      reinterpret_cast<unsigned char&>(compare),
                      reinterpret_cast<unsigned char&>(value)));
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
                      reinterpret_cast<uint16_t&>(value));
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
                           reinterpret_cast<uint16_t&>(value)));
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
                      reinterpret_cast<uint16_t&>(compare),
                      reinterpret_cast<uint16_t&>(value)));
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
                      reinterpret_cast<unsigned short&>(value));
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
                           reinterpret_cast<unsigned short&>(value)));
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
                      reinterpret_cast<unsigned short&>(compare),
                      reinterpret_cast<unsigned short&>(value)));
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
                      reinterpret_cast<uint32_t&>(value));
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
                           reinterpret_cast<uint32_t&>(value)));
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
                      reinterpret_cast<uint32_t&>(compare),
                      reinterpret_cast<uint32_t&>(value)));
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
                      reinterpret_cast<unsigned int&>(value));
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
                           reinterpret_cast<unsigned int&>(value)));
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
                      reinterpret_cast<unsigned int&>(compare),
                      reinterpret_cast<unsigned int&>(value)));
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
                      reinterpret_cast<uint64_t&>(value));
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
                           reinterpret_cast<uint64_t&>(value)));
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
                      reinterpret_cast<uint64_t&>(compare),
                      reinterpret_cast<uint64_t&>(value)));
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
                      reinterpret_cast<unsigned long long&>(value));
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
                           reinterpret_cast<unsigned long long&>(value)));
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
                      reinterpret_cast<unsigned long long&>(compare),
                      reinterpret_cast<unsigned long long&>(value)));
}

#endif

#endif  // RAJA_COMPILER_MSVC


template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE
    typename std::enable_if<sizeof(T) == sizeof(unsigned), T>::type
    builtin_atomic_load(T *acc)
{
  return RAJA::util::reinterp_A_as_B<unsigned, T>(
      builtin_atomic_load((unsigned *)acc));
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE
    typename std::enable_if<sizeof(T) == sizeof(unsigned long long), T>::type
    builtin_atomic_load(T *acc)
{
  return RAJA::util::reinterp_A_as_B<unsigned long long, T>(
      builtin_atomic_load((unsigned long long *)acc));
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE
    typename std::enable_if<sizeof(T) == sizeof(unsigned), void>::type
    builtin_atomic_store(T *acc, T value)
{
  builtin_atomic_store((unsigned *)acc, RAJA::util::reinterp_A_as_B<T, unsigned>(value));
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE
    typename std::enable_if<sizeof(T) == sizeof(unsigned long long), void>::type
    builtin_atomic_store(T *acc, T value)
{
  builtin_atomic_store((unsigned long long *)acc, RAJA::util::reinterp_A_as_B<T, unsigned long long>(value));
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE
    typename std::enable_if<sizeof(T) == sizeof(unsigned), T>::type
    builtin_atomic_CAS(T *acc, T compare, T value)
{
  return RAJA::util::reinterp_A_as_B<unsigned, T>(
      builtin_atomic_CAS((unsigned *)acc,
                         RAJA::util::reinterp_A_as_B<T, unsigned>(compare),
                         RAJA::util::reinterp_A_as_B<T, unsigned>(value)));
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE
    typename std::enable_if<sizeof(T) == sizeof(unsigned long long), T>::type
    builtin_atomic_CAS(T *acc, T compare, T value)
{
  return RAJA::util::reinterp_A_as_B<unsigned long long, T>(builtin_atomic_CAS(
      (unsigned long long *)acc,
      RAJA::util::reinterp_A_as_B<T, unsigned long long>(compare),
      RAJA::util::reinterp_A_as_B<T, unsigned long long>(value)));
}

/*!
 * Generic impementation of any atomic 32-bit or 64-bit operator that can be
 * implemented using a compare and swap primitive.
 * Implementation uses the builtin unsigned 32-bit and 64-bit CAS operators.
 * Returns the OLD value that was replaced by the result of this operation.
 */
template <typename T, typename OPER>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomic_CAS_oper(T *acc,
                                                      OPER &&oper)
{
  static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                "builtin atomic cas assumes 4 or 8 byte targets");

  BuiltinAtomicType<T> * accConverted = (BuiltinAtomicType<T> *) acc;
  BuiltinAtomicType<T> old = builtin_atomic_load(accConverted);
  BuiltinAtomicType<T> expected;

  do {
    expected = old;
    old = builtin_atomic_CAS(accConverted, expected, RAJA::util::reinterp_A_as_B<T, BuiltinAtomicType<T>>(oper(RAJA::util::reinterp_A_as_B<BuiltinAtomicType<T>, T>(expected))));
  } while (old != expected);

  return RAJA::util::reinterp_A_as_B<BuiltinAtomicType<T>, T>(old);
}

template <typename T, typename OPER, typename ShortCircuit>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomic_CAS_oper_sc(T *acc,
                                                         OPER &&oper,
                                                         ShortCircuit const &sc)
{
  static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                "builtin atomic cas assumes 4 or 8 byte targets");

  BuiltinAtomicType<T> * accConverted = (BuiltinAtomicType<T> *) acc;
  BuiltinAtomicType<T> old = builtin_atomic_load(accConverted);
  BuiltinAtomicType<T> expected;

  if (sc(RAJA::util::reinterp_A_as_B<BuiltinAtomicType<T>, T>(old))) {
    return RAJA::util::reinterp_A_as_B<BuiltinAtomicType<T>, T>(old);
  }

  do {
    expected = old;
    old = builtin_atomic_CAS(accConverted, expected, RAJA::util::reinterp_A_as_B<T, BuiltinAtomicType<T>>(oper(RAJA::util::reinterp_A_as_B<BuiltinAtomicType<T>, T>(expected))));
  } while (old != expected && !sc(RAJA::util::reinterp_A_as_B<BuiltinAtomicType<T>, T>(old)));

  return RAJA::util::reinterp_A_as_B<BuiltinAtomicType<T>, T>(old);
}


}  // namespace detail


template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicLoad(builtin_atomic,
                                         T *acc)
{
  return detail::builtin_atomic_load(acc);
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE void atomicStore(builtin_atomic,
                                             T *acc,
                                             T value)
{
  detail::builtin_atomic_store(acc, value);
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicAdd(builtin_atomic,
                                        T *acc,
                                        T value)
{
  return detail::builtin_atomic_CAS_oper(acc, [=](T a) { return a + value; });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicSub(builtin_atomic,
                                        T *acc,
                                        T value)
{
  return detail::builtin_atomic_CAS_oper(acc, [=](T a) { return a - value; });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicMin(builtin_atomic,
                                        T *acc,
                                        T value)
{
  return detail::builtin_atomic_CAS_oper_sc(acc,
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
  return detail::builtin_atomic_CAS_oper_sc(acc,
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
  return detail::builtin_atomic_CAS_oper(acc, [=](T a) { return a + 1; });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicInc(builtin_atomic, T *acc, T val)
{
  return detail::builtin_atomic_CAS_oper(acc, [=](T old) {
    return ((old >= val) ? 0 : (old + 1));
  });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicDec(builtin_atomic, T *acc)
{
  return detail::builtin_atomic_CAS_oper(acc, [=](T a) { return a - 1; });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicDec(builtin_atomic, T *acc, T val)
{
  return detail::builtin_atomic_CAS_oper(acc, [=](T old) {
    return (((old == 0) | (old > val)) ? val : (old - 1));
  });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicAnd(builtin_atomic,
                                        T *acc,
                                        T value)
{
  return detail::builtin_atomic_CAS_oper(acc, [=](T a) { return a & value; });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicOr(builtin_atomic, T *acc, T value)
{
  return detail::builtin_atomic_CAS_oper(acc, [=](T a) { return a | value; });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicXor(builtin_atomic,
                                        T *acc,
                                        T value)
{
  return detail::builtin_atomic_CAS_oper(acc, [=](T a) { return a ^ value; });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicExchange(builtin_atomic,
                                             T *acc,
                                             T value)
{
  return detail::builtin_atomic_CAS_oper(acc, [=](T) { return value; });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T
atomicCAS(builtin_atomic, T *acc, T compare, T value)
{
  return detail::builtin_atomic_CAS(acc, compare, value);
}


}  // namespace RAJA

// make sure this define doesn't bleed out of this header
#undef RAJA_AUTO_ATOMIC

#endif
