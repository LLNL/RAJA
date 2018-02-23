/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining atomic operations for ROCM
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_rocm_atomic_HPP
#define RAJA_policy_rocm_atomic_HPP

#include "RAJA/config.hpp"
#include "RAJA/util/Operators.hpp"
#include "RAJA/util/TypeConvert.hpp"
#include "RAJA/util/defines.hpp"

#include <stdexcept>

#if defined(RAJA_ENABLE_ROCM)
#include <hc.hpp>


namespace RAJA
{
namespace atomic
{

namespace detail
{

template <size_t BYTES>
struct ROCmAtomicCAS {
};



template <>
struct ROCmAtomicCAS<4> {

  /*!
   * Generic impementation of any atomic 32-bit operator.
   * Implementation uses the existing ROCm supplied unsigned 32-bit CAS
   * operator. Returns the OLD value that was replaced by the result of this
   * operation.
   */
  template <typename T, typename OPER>
  RAJA_INLINE RAJA_HOST_DEVICE T operator()(T volatile *acc, OPER const &oper) const 
  {
    // asserts in RAJA::util::reinterp_T_as_u and RAJA::util::reinterp_u_as_T
    // will enforce 32-bit T
    union U {
      unsigned i;
      T t;
      RAJA_INLINE U() {};
    } readback, oldval , newval ;

    oldval.t = *acc;
    readback.i = oldval.i;
    do {
       oldval.i = readback.i;
       newval.t = oper(oldval.t);
       hc::atomic_compare_exchange((unsigned *)acc, &readback.i, newval.i);

    } while (readback.i != oldval.i) ;

    return oldval.t;
  }
};

template <>
struct ROCmAtomicCAS<8> {

  /*!
   * Generic impementation of any atomic 64-bit operator.
   * Implementation uses the existing ROCM supplied unsigned 64-bit CAS
   * operator. Returns the OLD value that was replaced by the result of this
   * operation.
   */
  template <typename T, typename OPER>
  RAJA_INLINE RAJA_HOST_DEVICE T operator()(T volatile *acc, OPER const &oper) const
  {
    // asserts in RAJA::util::reinterp_T_as_u and RAJA::util::reinterp_u_as_T
    // will enforce 64-bit T
    union U {
      uint64_t i;
      T t;
      RAJA_INLINE U() {};
    } readback, oldval , newval ;

    oldval.t = *acc;
    readback.i = oldval.i;
    do {
       oldval.i = readback.i;
       newval.t = oper(oldval.t);
       hc::atomic_compare_exchange((uint64_t *)acc, &readback.i, newval.i);

    } while (readback.i != oldval.i) ;

    return oldval.t;
  }
};

/*!
 * Generic impementation of any atomic 32-bit or 64-bit operator that can be
 * implemented using a compare and swap primitive.
 * Implementation uses the existing ROCM supplied unsigned 32-bit and 64-bit
 * CAS operators.
 * Returns the OLD value that was replaced by the result of this operation.
 */
template <typename T, typename OPER>
RAJA_INLINE RAJA_HOST_DEVICE T rocm_atomic_CAS_oper(T volatile *acc, OPER &&oper)
{
  ROCmAtomicCAS<sizeof(T)> cas;
  return cas(acc, std::forward<OPER>(oper));
}
}  // namespace detail


struct rocm_atomic {
};


/*!
 * Catch-all policy passes off to ROCM's builtin atomics.
 *
 * This catch-all will only work for types supported by the compiler.
 * Specialization below can adapt for some unsupported types.
 */
template <typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicAdd(rocm_atomic, T volatile *acc, T value)
{
  return detail::rocm_atomic_CAS_oper(acc, [=] (T a) {
    return (a + value);
  });
}


// 32-bit signed atomicAdd support by ROCM
template <>
RAJA_INLINE RAJA_HOST_DEVICE int atomicAdd<int>(rocm_atomic, int volatile *acc,
                                int value)
{
  return hc::atomic_fetch_add((int *)acc, value);
}

// 32-bit unsigned atomicAdd support by ROCM
template <>
RAJA_INLINE RAJA_HOST_DEVICE unsigned atomicAdd<unsigned>(rocm_atomic,
                                                    unsigned volatile *acc,
                                                    unsigned value)
{
  return hc::atomic_fetch_add((unsigned *)acc, value);
}

// 64-bit unsigned atomicAdd support by ROCM
template <>
RAJA_INLINE RAJA_HOST_DEVICE unsigned long long atomicAdd<unsigned long long>(
    rocm_atomic,
    unsigned long long volatile *acc,
    unsigned long long value) 
{
  return hc::atomic_fetch_add((uint64_t *)acc, value);
}


// 32-bit float atomicAdd support by ROCM
template <>
RAJA_INLINE RAJA_HOST_DEVICE float atomicAdd<float>(rocm_atomic,
                                              float volatile *acc,
                                              float value)
{
  return hc::atomic_fetch_add((float *)acc, value);
}


// 64-bit double atomicAdd 
template <>
RAJA_INLINE RAJA_HOST_DEVICE double atomicAdd<double>(rocm_atomic,
                                                double volatile *acc,
                                                double value) 
{
#if 0
    union U {
      uint64_t i;
      double   t;
      RAJA_INLINE U() {};
    } readback, oldval , newval ;

    oldval.t = *acc;
    readback.i = oldval.i;
    do {
       oldval.i = readback.i;
       newval.t = oldval.t+value;
       hc::atomic_compare_exchange((uint64_t *)acc, &readback.i, newval.i);

    } while (readback.i != oldval.i) ;

    return oldval.t;
#else
  return detail::rocm_atomic_CAS_oper(acc, [=] (double a) {
    return (a + value);
  });
#endif
}


template <typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicSub(rocm_atomic, T volatile *acc, T value)
{
  return detail::rocm_atomic_CAS_oper(acc, [=] (T a) {
    return a - value;
  });
}

// 32-bit signed atomicSub support by ROCM
template <>
RAJA_INLINE RAJA_HOST_DEVICE int atomicSub<int>(rocm_atomic,
                                          int volatile *acc,
                                          int value)
{
  return hc::atomic_fetch_sub((int *)acc, value);
}


// 32-bit unsigned atomicSub support by ROCM
template <>
RAJA_INLINE RAJA_HOST_DEVICE unsigned atomicSub<unsigned>(rocm_atomic,
                                                    unsigned volatile *acc,
                                                    unsigned value)
{
  return hc::atomic_fetch_sub((unsigned *)acc, value);
}


template <typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicMin(rocm_atomic, T volatile *acc, T value)
{
  return detail::rocm_atomic_CAS_oper(acc, [=] (T a) {
    return a < value ? a : value;
  });
}


// 32-bit signed atomicMin support by ROCM
template <>
RAJA_INLINE RAJA_HOST_DEVICE int atomicMin<int>(rocm_atomic,
                                          int volatile *acc,
                                          int value)
{
  return hc::atomic_fetch_min((int *)acc, value);
}


// 32-bit unsigned atomicMin support by ROCM
template <>
RAJA_INLINE RAJA_HOST_DEVICE unsigned atomicMin<unsigned>(rocm_atomic,
                                                    unsigned volatile *acc,
                                                    unsigned value)
{
  return hc::atomic_fetch_min((unsigned *)acc, value);
}

// 64-bit unsigned atomicMin support by ROCM 
template <>
RAJA_INLINE RAJA_HOST_DEVICE unsigned long long atomicMin<unsigned long long>(
    rocm_atomic,
    unsigned long long volatile *acc,
    unsigned long long value)
{
  return hc::atomic_fetch_min((uint64_t *)acc, (uint64_t)value);
}


template <typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicMax(rocm_atomic, T volatile *acc, T value)
{
  return detail::rocm_atomic_CAS_oper(acc, [=] (T a) {
    return a > value ? a : value;
  });
}

// 32-bit signed atomicMax support by ROCM
template <>
RAJA_INLINE RAJA_HOST_DEVICE int atomicMax<int>(rocm_atomic,
                                          int volatile *acc,
                                          int value)
{
  return hc::atomic_fetch_max((int *)acc, value);
}


// 32-bit unsigned atomicMax support by ROCM
template <>
RAJA_INLINE RAJA_HOST_DEVICE unsigned atomicMax<unsigned>(rocm_atomic,
                                                    unsigned volatile *acc,
                                                    unsigned value)
{
  return hc::atomic_fetch_max((unsigned *)acc, value);
}

// 64-bit unsigned atomicMax support by ROCM
template <>
RAJA_INLINE RAJA_HOST_DEVICE unsigned long long atomicMax<unsigned long long>(
    rocm_atomic,
    unsigned long long volatile *acc,
    unsigned long long value)
{
  return hc::atomic_fetch_max((uint64_t *)acc, value);
}


template <typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicInc(rocm_atomic, T volatile *acc, T val)
{
  return detail::rocm_atomic_CAS_oper(acc, [=] (T old) {
    return ((old >= val) ? 0 : (old + 1));
  });
}

// 32-bit unsigned atomicInc support by ROCM
template <>
RAJA_INLINE RAJA_HOST_DEVICE unsigned atomicInc<unsigned>(rocm_atomic,
                                                    unsigned volatile *acc,
                                                    unsigned value)
{
  return detail::rocm_atomic_CAS_oper(acc, [=] (unsigned old) {
    return ((old >= value) ? 0 : (old + 1));
  });
}


template <typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicInc(rocm_atomic, T volatile *acc)
{
  return detail::rocm_atomic_CAS_oper(acc,
                                      [=] (T a) { return a + 1; });
}


template <typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicDec(rocm_atomic, T volatile *acc, T val)
{
  return detail::rocm_atomic_CAS_oper(acc, [=] (T old) {
    return (((old == 0) | (old > val)) ? val : (old - 1));
  });
}

// 32-bit unsigned atomicDec support by ROCM
template <>
RAJA_INLINE RAJA_HOST_DEVICE unsigned atomicDec<unsigned>(rocm_atomic,
                                                    unsigned volatile *acc,
                                                    unsigned value)
{
  return detail::rocm_atomic_CAS_oper(acc, [=] (unsigned old) {
    return (((old == 0) | (old > value)) ? value : (old - 1));
  });
}

template <typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicDec(rocm_atomic, T volatile *acc)
{
  return detail::rocm_atomic_CAS_oper(acc,
                                      [=] (T a) { return a - 1; });
}


template <typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicAnd(rocm_atomic, T volatile *acc, T value)
{
  return detail::rocm_atomic_CAS_oper(acc, [=] (T a) {
    return a & value;
  });
}

// 32-bit signed atomicAnd support by ROCM
template <>
RAJA_INLINE RAJA_HOST_DEVICE int atomicAnd<int>(rocm_atomic,
                                          int volatile *acc,
                                          int value)
{
  return hc::atomic_fetch_and((int *)acc, value);
}


// 32-bit unsigned atomicAnd support by ROCM
template <>
RAJA_INLINE RAJA_HOST_DEVICE unsigned atomicAnd<unsigned>(rocm_atomic,
                                                    unsigned volatile *acc,
                                                    unsigned value)
{
  return hc::atomic_fetch_and((unsigned *)acc, value);
}

// 64-bit unsigned atomicAnd support by ROCM
template <>
RAJA_INLINE RAJA_HOST_DEVICE unsigned long long atomicAnd<unsigned long long>(
    rocm_atomic,
    unsigned long long volatile *acc,
    unsigned long long value)
{
  return hc::atomic_fetch_and((uint64_t *)acc, value);
}


template <typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicOr(rocm_atomic, T volatile *acc, T value) 
{
  return detail::rocm_atomic_CAS_oper(acc, [=] (T a) {
    return a | value;
  });
}

// 32-bit signed atomicOr 
template <>
RAJA_INLINE RAJA_HOST_DEVICE int atomicOr<int>(rocm_atomic,
                                         int volatile *acc,
                                         int value)
{
  return hc::atomic_fetch_or((int *)acc, value);
}


// 32-bit unsigned atomicOr 
template <>
RAJA_INLINE RAJA_HOST_DEVICE unsigned atomicOr<unsigned>(rocm_atomic,
                                                   unsigned volatile *acc,
                                                   unsigned value)
{
  return hc::atomic_fetch_or((unsigned *)acc, value);
}

template <>
RAJA_INLINE RAJA_HOST_DEVICE unsigned long long atomicOr<unsigned long long>(
    rocm_atomic,
    unsigned long long volatile *acc,
    unsigned long long value)
{
  return hc::atomic_fetch_or((uint64_t *)acc, value);
}


template <typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicXor(rocm_atomic, T volatile *acc, T value)
{
  return detail::rocm_atomic_CAS_oper(acc, [=] (T a) {
    return a ^ value;
  });
}

// 32-bit signed atomicXor
template <>
RAJA_INLINE RAJA_HOST_DEVICE int atomicXor<int>(rocm_atomic,
                                          int volatile *acc,
                                          int value)
{
  return hc::atomic_fetch_xor((int *)acc, value);
}


// 32-bit unsigned atomicXor
template <>
RAJA_INLINE RAJA_HOST_DEVICE unsigned atomicXor<unsigned>(rocm_atomic,
                                                    unsigned volatile *acc,
                                                    unsigned value)
{
  return hc::atomic_fetch_xor((unsigned *)acc, value);
}

template <>
RAJA_INLINE RAJA_HOST_DEVICE unsigned long long atomicXor<unsigned long long>(
    rocm_atomic,
    unsigned long long volatile *acc,
    unsigned long long value)
{
  return hc::atomic_fetch_xor((uint64_t *)acc, value);
}



template <typename T>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicExchange(rocm_atomic, T volatile *acc, T compare,
          typename std::enable_if<sizeof(T)==sizeof(unsigned), T>::type value)
{
  // attempt to use the ROCm builtin atomic, if it exists for T
  return hc::atomic_compare_exchange((unsigned *)acc, &compare, value);
}

template <typename T>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicExchange(rocm_atomic, T volatile *acc, T compare,
          typename std::enable_if<sizeof(T)==sizeof(uint64_t), T>::type value)
{
  // attempt to use the ROCm builtin atomic, if it exists for T
  return hc::atomic_compare_exchange((uint64_t*)acc, &compare, value);
}

template <typename T>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicCAS(rocm_atomic, T volatile *acc, T compare, 
          typename std::enable_if<sizeof(T)==sizeof(unsigned), T>::type value)
{
  // attempt to use the ROCm builtin atomic, if it exists for T
  return hc::atomic_compare_exchange((unsigned *)acc, &compare, value);
}

template <typename T>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicCAS(rocm_atomic, T volatile *acc, T compare, 
          typename std::enable_if<sizeof(T)==sizeof(uint64_t), T>::type value)
{
  // attempt to use the ROCm builtin atomic, if it exists for T
  return hc::atomic_compare_exchange((uint64_t*)acc, &compare, value);
}


}  // namespace atomic
}  // namespace RAJA


#endif  // RAJA_ENABLE_ROCM
#endif  // guard
