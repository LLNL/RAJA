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


#if defined(__HCC_ACCELERATOR__ )
//#if __KALMAR_ACCELERATOR__ == 1
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
  RAJA_INLINE T operator()(T volatile *acc, OPER const &oper) const [[hc]]
  {
    // asserts in RAJA::util::reinterp_T_as_u and RAJA::util::reinterp_u_as_T
    // will enforce 32-bit T
    unsigned oldval, newval, readback;
    oldval = RAJA::util::reinterp_A_as_B<T, unsigned>(*acc);
    newval = RAJA::util::reinterp_A_as_B<T, unsigned>(
        oper(RAJA::util::reinterp_A_as_B<unsigned, T>(oldval)));
    while ((readback = hc::atomic_compare_exchange_unsigned((unsigned *)acc, oldval, newval))
           != oldval) {
      oldval = readback;
      newval = RAJA::util::reinterp_A_as_B<T, unsigned>(
          oper(RAJA::util::reinterp_A_as_B<unsigned, T>(oldval)));
    }
    return RAJA::util::reinterp_A_as_B<unsigned, T>(oldval);
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
  RAJA_INLINE T operator()(T volatile *acc, OPER const &oper) const [[hc]]
  {
    // asserts in RAJA::util::reinterp_T_as_u and RAJA::util::reinterp_u_as_T
    // will enforce 64-bit T
    uint64_t oldval, newval, readback;
    oldval = RAJA::util::reinterp_A_as_B<T, uint64_t>(*acc);
    newval = RAJA::util::reinterp_A_as_B<T, uint64_t>(
        oper(RAJA::util::reinterp_A_as_B<uint64_t, T>(oldval)));
    while ((readback = hc::atomic_compare_exchange_uint64((uint64_t *)acc, oldval, newval))
           != oldval) {
      oldval = readback;
      newval = RAJA::util::reinterp_A_as_B<T, uint64_t>(
          oper(RAJA::util::reinterp_A_as_B<uint64_t, T>(oldval)));
    }
    return RAJA::util::reinterp_A_as_B<uint64_t, T>(oldval);
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
RAJA_INLINE T rocm_atomic_CAS_oper(T volatile *acc, OPER &&oper) [[hc]]
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
RAJA_INLINE T atomicAdd(rocm_atomic, T volatile *acc, T value) [[hc]]
{
  return detail::rocm_atomic_CAS_oper(acc, [=] (T a) [[hc]] {
    return a + value;
  });
}


// 32-bit signed atomicAdd support by ROCM
template <>
RAJA_INLINE int atomicAdd<int>(rocm_atomic, int volatile *acc,
                                int value) [[hc]]
{
  return hc::atomic_add_int((int *)acc, value);
}


// 32-bit unsigned atomicAdd support by ROCM
template <>
RAJA_INLINE unsigned atomicAdd<unsigned>(rocm_atomic,
                                                    unsigned volatile *acc,
                                                    unsigned value) [[hc]]
{
  return hc::atomic_add_unsigned((unsigned *)acc, value);
}

// 64-bit unsigned atomicAdd support by ROCM
template <>
RAJA_INLINE unsigned long long atomicAdd<unsigned long long>(
    rocm_atomic,
    unsigned long long volatile *acc,
    unsigned long long value) [[hc]]
{
  return hc::atomic_add_uint64((uint64_t *)acc, value);
}


// 32-bit float atomicAdd support by ROCM
template <>
RAJA_INLINE float atomicAdd<float>(rocm_atomic,
                                              float volatile *acc,
                                              float value) [[hc]]
{
  return hc::atomic_add_float((float *)acc, value);
}


// 64-bit double atomicAdd 
template <>
RAJA_INLINE double atomicAdd<double>(rocm_atomic,
                                                double volatile *acc,
                                                double value) [[hc]]
{
    union U {
      uint64_t i;
      double   t;
      RAJA_INLINE U() {};
    } readback, oldval , newval ;

    oldval.t = *acc ;

    do {
      readback.i = oldval.i ;
      newval.t = readback.t + value ;
      oldval.i = hc::atomic_compare_exchange_uint64((uint64_t *)acc, readback.i , newval.i );
    } while ( readback.i != oldval.i );
    return oldval.t ;
}


template <typename T>
RAJA_INLINE T atomicSub(rocm_atomic, T volatile *acc, T value) [[hc]]
{
  return detail::rocm_atomic_CAS_oper(acc, [=] (T a) {
    return a - value;
  });
}

// 32-bit signed atomicSub support by ROCM
template <>
RAJA_INLINE int atomicSub<int>(rocm_atomic,
                                          int volatile *acc,
                                          int value) [[hc]]
{
  return hc::atomic_sub_int((int *)acc, value);
}


// 32-bit unsigned atomicSub support by ROCM
template <>
RAJA_INLINE unsigned atomicSub<unsigned>(rocm_atomic,
                                                    unsigned volatile *acc,
                                                    unsigned value) [[hc]]
{
  return hc::atomic_sub_unsigned((unsigned *)acc, value);
}


template <typename T>
RAJA_INLINE T atomicMin(rocm_atomic, T volatile *acc, T value) [[hc]]
{
  return detail::rocm_atomic_CAS_oper(acc, [=] (T a) {
    return a < value ? a : value;
  });
}


// 32-bit signed atomicMin support by ROCM
template <>
RAJA_INLINE int atomicMin<int>(rocm_atomic,
                                          int volatile *acc,
                                          int value) [[hc]]
{
  return hc::atomic_min_int((int *)acc, value);
}


// 32-bit unsigned atomicMin support by ROCM
template <>
RAJA_INLINE unsigned atomicMin<unsigned>(rocm_atomic,
                                                    unsigned volatile *acc,
                                                    unsigned value) [[hc]]
{
  return hc::atomic_min_unsigned((unsigned *)acc, value);
}

// 64-bit unsigned atomicMin support by ROCM 
template <>
RAJA_INLINE unsigned long long atomicMin<unsigned long long>(
    rocm_atomic,
    unsigned long long volatile *acc,
    unsigned long long value) [[hc]]
{
  return hc::atomic_fetch_min((uint64_t *)acc, (uint64_t)value);
}


template <typename T>
RAJA_INLINE T atomicMax(rocm_atomic, T volatile *acc, T value) [[hc]]
{
  return detail::rocm_atomic_CAS_oper(acc, [=] (T a) {
    return a > value ? a : value;
  });
}

// 32-bit signed atomicMax support by ROCM
template <>
RAJA_INLINE int atomicMax<int>(rocm_atomic,
                                          int volatile *acc,
                                          int value) [[hc]]
{
  return hc::atomic_max_int((int *)acc, value);
}


// 32-bit unsigned atomicMax support by ROCM
template <>
RAJA_INLINE unsigned atomicMax<unsigned>(rocm_atomic,
                                                    unsigned volatile *acc,
                                                    unsigned value) [[hc]]
{
  return hc::atomic_max_unsigned((unsigned *)acc, value);
}

// 64-bit unsigned atomicMax support by ROCM
template <>
RAJA_INLINE unsigned long long atomicMax<unsigned long long>(
    rocm_atomic,
    unsigned long long volatile *acc,
    unsigned long long value) [[hc]]
{
  return hc::atomic_max_uint64((uint64_t *)acc, value);
}


template <typename T>
RAJA_INLINE T atomicInc(rocm_atomic, T volatile *acc, T val) [[hc]]
{
  return detail::rocm_atomic_CAS_oper(acc, [=] (T old) {
    return ((old >= val) ? 0 : (old + 1));
  });
}

/*
// 32-bit unsigned atomicInc support by ROCM
template <>
RAJA_INLINE unsigned atomicInc<unsigned>(rocm_atomic,
                                                    unsigned volatile *acc,
                                                    unsigned value) [[hc]]
{
  return hc::atomic_inc_unsigned((unsigned *)acc);
}
*/


template <typename T>
RAJA_INLINE T atomicInc(rocm_atomic, T volatile *acc) [[hc]]
{
  return detail::rocm_atomic_CAS_oper(acc,
                                      [=] (T a) { return a + 1; });
}


template <typename T>
RAJA_INLINE T atomicDec(rocm_atomic, T volatile *acc, T val) [[hc]]
{
  return detail::rocm_atomic_CAS_oper(acc, [=] (T old) {
    return (((old == 0) | (old > val)) ? val : (old - 1));
  });
}

/*
// 32-bit unsigned atomicDec support by ROCM
template <>
RAJA_INLINE unsigned atomicDec<unsigned>(rocm_atomic,
                                                    unsigned volatile *acc,
                                                    unsigned value) [[hc]]
{
  return hc::atomic_dec_unsigned((unsigned *)acc);
}
*/

template <typename T>
RAJA_INLINE T atomicDec(rocm_atomic, T volatile *acc) [[hc]]
{
  return detail::rocm_atomic_CAS_oper(acc,
                                      [=] (T a) { return a - 1; });
}


template <typename T>
RAJA_INLINE T atomicAnd(rocm_atomic, T volatile *acc, T value) [[hc]]
{
  return detail::rocm_atomic_CAS_oper(acc, [=] (T a) {
    return a & value;
  });
}

// 32-bit signed atomicAnd support by ROCM
template <>
RAJA_INLINE int atomicAnd<int>(rocm_atomic,
                                          int volatile *acc,
                                          int value) [[hc]]
{
  return hc::atomic_and_int((int *)acc, value);
}


// 32-bit unsigned atomicAnd support by ROCM
template <>
RAJA_INLINE unsigned atomicAnd<unsigned>(rocm_atomic,
                                                    unsigned volatile *acc,
                                                    unsigned value) [[hc]]
{
  return hc::atomic_and_unsigned((unsigned *)acc, value);
}

// 64-bit unsigned atomicAnd support by ROCM
template <>
RAJA_INLINE unsigned long long atomicAnd<unsigned long long>(
    rocm_atomic,
    unsigned long long volatile *acc,
    unsigned long long value) [[hc]]
{
  return hc::atomic_and_uint64((uint64_t *)acc, value);
}


template <typename T>
RAJA_INLINE T atomicOr(rocm_atomic, T volatile *acc, T value)  [[hc]]
{
  return detail::rocm_atomic_CAS_oper(acc, [=] (T a) {
    return a | value;
  });
}

// 32-bit signed atomicOr 
template <>
RAJA_INLINE int atomicOr<int>(rocm_atomic,
                                         int volatile *acc,
                                         int value) [[hc]]
{
  return hc::atomic_or_int((int *)acc, value);
}


// 32-bit unsigned atomicOr 
template <>
RAJA_INLINE unsigned atomicOr<unsigned>(rocm_atomic,
                                                   unsigned volatile *acc,
                                                   unsigned value) [[hc]]
{
  return hc::atomic_or_unsigned((unsigned *)acc, value);
}

template <>
RAJA_INLINE unsigned long long atomicOr<unsigned long long>(
    rocm_atomic,
    unsigned long long volatile *acc,
    unsigned long long value) [[hc]]
{
  return hc::atomic_or_uint64((uint64_t *)acc, value);
}


template <typename T>
RAJA_INLINE T atomicXor(rocm_atomic, T volatile *acc, T value) [[hc]]
{
  return detail::rocm_atomic_CAS_oper(acc, [=] (T a) {
    return a ^ value;
  });
}

// 32-bit signed atomicXor
template <>
RAJA_INLINE int atomicXor<int>(rocm_atomic,
                                          int volatile *acc,
                                          int value) [[hc]]
{
  return hc::atomic_xor_int((int *)acc, value);
}


// 32-bit unsigned atomicXor
template <>
RAJA_INLINE unsigned atomicXor<unsigned>(rocm_atomic,
                                                    unsigned volatile *acc,
                                                    unsigned value) [[hc]]
{
  return hc::atomic_xor_unsigned((unsigned *)acc, value);
}

template <>
RAJA_INLINE unsigned long long atomicXor<unsigned long long>(
    rocm_atomic,
    unsigned long long volatile *acc,
    unsigned long long value) [[hc]]
{
  return hc::atomic_xor_uint64((uint64_t *)acc, value);
}



template <typename T>
RAJA_INLINE T
atomicExchange(rocm_atomic, T volatile *acc, T compare,
          typename std::enable_if<sizeof(T)==sizeof(unsigned), T>::type value)
          [[hc]]
{
  // attempt to use the ROCm builtin atomic, if it exists for T
  return hc::atomic_exchange_unsigned((unsigned *)acc, compare, value);
}

template <typename T>
RAJA_INLINE T
atomicExchange(rocm_atomic, T volatile *acc, T compare,
          typename std::enable_if<sizeof(T)==sizeof(uint64_t), T>::type value)
          [[hc]]
{
  // attempt to use the ROCm builtin atomic, if it exists for T
  return hc::atomic_exchange_uint64((uint64_t*)acc, compare, value);
}

template <typename T>
RAJA_INLINE T
atomicCAS(rocm_atomic, T volatile *acc, T compare, 
          typename std::enable_if<sizeof(T)==sizeof(unsigned), T>::type value)
          [[hc]]
{
  // attempt to use the ROCm builtin atomic, if it exists for T
  return hc::atomic_compare_exchange_unsigned((unsigned *)acc, compare, value);
}

template <typename T>
RAJA_INLINE T
atomicCAS(rocm_atomic, T volatile *acc, T compare, 
          typename std::enable_if<sizeof(T)==sizeof(uint64_t), T>::type value)
          [[hc]]
{
  // attempt to use the ROCm builtin atomic, if it exists for T
  return hc::atomic_compare_exchange_uint64((uint64_t*)acc, compare, value);
}


}  // namespace atomic
}  // namespace RAJA
#endif  // __HCC_ACCELERATOR__


#endif  // RAJA_ENABLE_ROCM
#endif  // guard
