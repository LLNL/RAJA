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
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_atomic_builtin_HPP
#define RAJA_policy_atomic_builtin_HPP

#include "RAJA/config.hpp"

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

namespace detail
{

#if defined(RAJA_COMPILER_MSVC) || (defined(_WIN32) && defined(__INTEL_COMPILER))

RAJA_DEVICE_HIP
RAJA_INLINE unsigned builtin_atomic_CAS(unsigned volatile *acc,
                                        unsigned compare,
                                        unsigned value)
{

  long long_value = RAJA::util::reinterp_A_as_B<unsigned, long>(value);
  long long_compare = RAJA::util::reinterp_A_as_B<unsigned, long>(compare);

  long old = _InterlockedCompareExchange((long *)acc, long_value, long_compare);

  return RAJA::util::reinterp_A_as_B<long, unsigned>(old);
}

RAJA_DEVICE_HIP
RAJA_INLINE unsigned long long builtin_atomic_CAS(
    unsigned long long volatile *acc,
    unsigned long long compare,
    unsigned long long value)
{

  long long long_value =
      RAJA::util::reinterp_A_as_B<unsigned long long, long long>(value);
  long long long_compare =
      RAJA::util::reinterp_A_as_B<unsigned long long, long long>(compare);

  long long old = _InterlockedCompareExchange64((long long volatile *)acc,
                                                long_value,
                                                long_compare);

  return RAJA::util::reinterp_A_as_B<long long, unsigned long long>(old);
}

#else  // RAJA_COMPILER_MSVC

RAJA_DEVICE_HIP
RAJA_INLINE unsigned builtin_atomic_CAS(unsigned volatile *acc,
                                        unsigned compare,
                                        unsigned value)
{
  __atomic_compare_exchange_n(
      acc, &compare, value, false, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED);
  return compare;
}

RAJA_DEVICE_HIP
RAJA_INLINE unsigned long long builtin_atomic_CAS(
    unsigned long long volatile *acc,
    unsigned long long compare,
    unsigned long long value)
{
  __atomic_compare_exchange_n(
      acc, &compare, value, false, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED);
  return compare;
}

#endif  // RAJA_COMPILER_MSVC


template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE
    typename std::enable_if<sizeof(T) == sizeof(unsigned), T>::type
    builtin_atomic_CAS(T volatile *acc, T compare, T value)
{
  return RAJA::util::reinterp_A_as_B<unsigned, T>(
      builtin_atomic_CAS((unsigned volatile *)acc,
                         RAJA::util::reinterp_A_as_B<T, unsigned>(compare),
                         RAJA::util::reinterp_A_as_B<T, unsigned>(value)));
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE
    typename std::enable_if<sizeof(T) == sizeof(unsigned long long), T>::type
    builtin_atomic_CAS(T volatile *acc, T compare, T value)
{
  return RAJA::util::reinterp_A_as_B<unsigned long long, T>(builtin_atomic_CAS(
      (unsigned long long volatile *)acc,
      RAJA::util::reinterp_A_as_B<T, unsigned long long>(compare),
      RAJA::util::reinterp_A_as_B<T, unsigned long long>(value)));
}


template <size_t BYTES>
struct BuiltinAtomicCAS;
template <size_t BYTES>
struct BuiltinAtomicCAS {
  static_assert(!(BYTES == 4 || BYTES == 8),
                "builtin atomic cas assumes 4 or 8 byte targets");
};


template <>
struct BuiltinAtomicCAS<4> {

  /*!
   * Generic impementation of any atomic 32-bit operator.
   * Implementation uses the existing builtin unsigned 32-bit CAS operator.
   * Returns the OLD value that was replaced by the result of this operation.
   */
  template <typename T, typename OPER, typename ShortCircuit>
  RAJA_DEVICE_HIP RAJA_INLINE T operator()(T volatile *acc,
                                           OPER const &oper,
                                           ShortCircuit const &sc) const
  {
#ifdef RAJA_COMPILER_MSVC
#pragma warning( disable : 4244 )  // Force msvc to not emit conversion warning
#endif
    unsigned oldval, newval, readback;

    oldval = RAJA::util::reinterp_A_as_B<T, unsigned>(*acc);
    newval = RAJA::util::reinterp_A_as_B<T, unsigned>(
        oper(RAJA::util::reinterp_A_as_B<unsigned, T>(oldval)));

    while ((readback = builtin_atomic_CAS((unsigned *)acc, oldval, newval)) !=
           oldval) {
      if (sc(readback)) break;
      oldval = readback;
      newval = RAJA::util::reinterp_A_as_B<T, unsigned>(
          oper(RAJA::util::reinterp_A_as_B<unsigned, T>(oldval)));
    }
    return RAJA::util::reinterp_A_as_B<unsigned, T>(oldval);
  }
#ifdef RAJA_COMPILER_MSVC
#pragma warning( default : 4244 )  // Reenable warning
#endif
};

template <>
struct BuiltinAtomicCAS<8> {

  /*!
   * Generic impementation of any atomic 64-bit operator.
   * Implementation uses the existing builtin unsigned 64-bit CAS operator.
   * Returns the OLD value that was replaced by the result of this operation.
   */
  template <typename T, typename OPER, typename ShortCircuit>
  RAJA_DEVICE_HIP RAJA_INLINE T operator()(T volatile *acc,
                                           OPER const &oper,
                                           ShortCircuit const &sc) const
  {
#ifdef RAJA_COMPILER_MSVC
#pragma warning( disable : 4244 )  // Force msvc to not emit conversion warning
#endif
    unsigned long long oldval, newval, readback;

    oldval = RAJA::util::reinterp_A_as_B<T, unsigned long long>(*acc);
    newval = RAJA::util::reinterp_A_as_B<T, unsigned long long>(
        oper(RAJA::util::reinterp_A_as_B<unsigned long long, T>(oldval)));

    while ((readback = builtin_atomic_CAS((unsigned long long *)acc,
                                          oldval,
                                          newval)) != oldval) {
      if (sc(readback)) break;
      oldval = readback;
      newval = RAJA::util::reinterp_A_as_B<T, unsigned long long>(
          oper(RAJA::util::reinterp_A_as_B<unsigned long long, T>(oldval)));
    }
    return RAJA::util::reinterp_A_as_B<unsigned long long, T>(oldval);
  }

#ifdef RAJA_COMPILER_MSVC
#pragma warning( default : 4244 )  // Reenable warning
#endif

};


/*!
 * Generic impementation of any atomic 32-bit or 64-bit operator that can be
 * implemented using a compare and swap primitive.
 * Implementation uses the builtin unsigned 32-bit and 64-bit CAS operators.
 * Returns the OLD value that was replaced by the result of this operation.
 */
template <typename T, typename OPER>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomic_CAS_oper(T volatile *acc,
                                                      OPER &&oper)
{
  BuiltinAtomicCAS<sizeof(T)> cas;
  return cas(acc, std::forward<OPER>(oper), [](T const &) { return false; });
}

template <typename T, typename OPER, typename ShortCircuit>
RAJA_DEVICE_HIP RAJA_INLINE T builtin_atomic_CAS_oper_sc(T volatile *acc,
                                                         OPER &&oper,
                                                         ShortCircuit const &sc)
{
  BuiltinAtomicCAS<sizeof(T)> cas;
  return cas(acc, std::forward<OPER>(oper), sc);
}


}  // namespace detail


template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicAdd(builtin_atomic,
                                        T volatile *acc,
                                        T value)
{
  return detail::builtin_atomic_CAS_oper(acc, [=](T a) { return a + value; });
}


template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicSub(builtin_atomic,
                                        T volatile *acc,
                                        T value)
{
  return detail::builtin_atomic_CAS_oper(acc, [=](T a) { return a - value; });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicMin(builtin_atomic,
                                        T volatile *acc,
                                        T value)
{
  if (*acc < value) {
    return *acc;
  }
  return detail::builtin_atomic_CAS_oper_sc(acc,
                                            [=](T a) {
                                              return a < value ? a : value;
                                            },
                                            [=](T current) {
                                              return current < value;
                                            });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicMax(builtin_atomic,
                                        T volatile *acc,
                                        T value)
{
  if (*acc > value) {
    return *acc;
  }
  return detail::builtin_atomic_CAS_oper_sc(acc,
                                            [=](T a) {
                                              return a > value ? a : value;
                                            },
                                            [=](T current) {
                                              return current > value;
                                            });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicInc(builtin_atomic, T volatile *acc)
{
  return detail::builtin_atomic_CAS_oper(acc, [=](T a) { return a + 1; });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicInc(builtin_atomic, T volatile *acc, T val)
{
  return detail::builtin_atomic_CAS_oper(acc, [=](T old) {
    return ((old >= val) ? 0 : (old + 1));
  });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicDec(builtin_atomic, T volatile *acc)
{
  return detail::builtin_atomic_CAS_oper(acc, [=](T a) { return a - 1; });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicDec(builtin_atomic, T volatile *acc, T val)
{
  return detail::builtin_atomic_CAS_oper(acc, [=](T old) {
    return (((old == 0) | (old > val)) ? val : (old - 1));
  });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicAnd(builtin_atomic,
                                        T volatile *acc,
                                        T value)
{
  return detail::builtin_atomic_CAS_oper(acc, [=](T a) { return a & value; });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicOr(builtin_atomic, T volatile *acc, T value)
{
  return detail::builtin_atomic_CAS_oper(acc, [=](T a) { return a | value; });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicXor(builtin_atomic,
                                        T volatile *acc,
                                        T value)
{
  return detail::builtin_atomic_CAS_oper(acc, [=](T a) { return a ^ value; });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T atomicExchange(builtin_atomic,
                                             T volatile *acc,
                                             T value)
{
  return detail::builtin_atomic_CAS_oper(acc, [=](T) { return value; });
}

template <typename T>
RAJA_DEVICE_HIP RAJA_INLINE T
atomicCAS(builtin_atomic, T volatile *acc, T compare, T value)
{
  return detail::builtin_atomic_CAS(acc, compare, value);
}


}  // namespace RAJA

// make sure this define doesn't bleed out of this header
#undef RAJA_AUTO_ATOMIC

#endif
