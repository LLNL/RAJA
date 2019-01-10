/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining atomic operations for CUDA
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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

#ifndef RAJA_policy_cuda_atomic_HPP
#define RAJA_policy_cuda_atomic_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include <stdexcept>
#include <type_traits>

#include "RAJA/util/Operators.hpp"
#include "RAJA/util/TypeConvert.hpp"
#include "RAJA/util/macros.hpp"


namespace RAJA
{
namespace atomic
{


namespace detail
{

/*!
 * Generic impementation of atomic 32-bit or 64-bit compare and swap primitive.
 * Implementation uses the existing CUDA supplied unsigned 32-bit and 64-bit
 * CAS operators.
 * Returns the value that was stored before this operation.
 */
RAJA_INLINE __device__ unsigned cuda_atomic_CAS(
    unsigned volatile *acc,
    unsigned compare,
    unsigned value)
{
  return ::atomicCAS((unsigned *)acc, compare, value);
}
///
RAJA_INLINE __device__ unsigned long long cuda_atomic_CAS(
    unsigned long long volatile *acc,
    unsigned long long compare,
    unsigned long long value)
{
  return ::atomicCAS((unsigned long long *)acc, compare, value);
}
///
template <typename T>
RAJA_INLINE __device__
typename std::enable_if<sizeof(T) == sizeof(unsigned), T>::type
cuda_atomic_CAS(T volatile *acc, T compare, T value)
{
  return RAJA::util::reinterp_A_as_B<unsigned, T>(
      cuda_atomic_CAS((unsigned volatile *)acc,
          RAJA::util::reinterp_A_as_B<T, unsigned>(compare),
          RAJA::util::reinterp_A_as_B<T, unsigned>(value)));
}
///
template <typename T>
RAJA_INLINE __device__
typename std::enable_if<sizeof(T) == sizeof(unsigned long long), T>::type
cuda_atomic_CAS(T volatile *acc, T compare, T value)
{
  return RAJA::util::reinterp_A_as_B<unsigned long long, T>(
      cuda_atomic_CAS((unsigned long long volatile *)acc,
          RAJA::util::reinterp_A_as_B<T, unsigned long long>(compare),
          RAJA::util::reinterp_A_as_B<T, unsigned long long>(value)));
}

template <size_t BYTES>
struct CudaAtomicCAS {
};


template <>
struct CudaAtomicCAS<4> {

  /*!
   * Generic impementation of any atomic 32-bit operator.
   * Implementation uses the existing CUDA supplied unsigned 32-bit CAS
   * operator. Returns the OLD value that was replaced by the result of this
   * operation.
   */
  template <typename T, typename OPER>
  RAJA_INLINE __device__ T operator()(T volatile *acc, OPER const &oper) const
  {
    // asserts in RAJA::util::reinterp_T_as_u and RAJA::util::reinterp_u_as_T
    // will enforce 32-bit T
    unsigned oldval, newval, readback;
    oldval = RAJA::util::reinterp_A_as_B<T, unsigned>(*acc);
    newval = RAJA::util::reinterp_A_as_B<T, unsigned>(
        oper(RAJA::util::reinterp_A_as_B<unsigned, T>(oldval)));
    while ((readback = cuda_atomic_CAS((unsigned volatile*)acc, oldval, newval)) !=
           oldval) {
      oldval = readback;
      newval = RAJA::util::reinterp_A_as_B<T, unsigned>(
          oper(RAJA::util::reinterp_A_as_B<unsigned, T>(oldval)));
    }
    return RAJA::util::reinterp_A_as_B<unsigned, T>(oldval);
  }
};

template <>
struct CudaAtomicCAS<8> {

  /*!
   * Generic impementation of any atomic 64-bit operator.
   * Implementation uses the existing CUDA supplied unsigned 64-bit CAS
   * operator. Returns the OLD value that was replaced by the result of this
   * operation.
   */
  template <typename T, typename OPER>
  RAJA_INLINE __device__ T operator()(T volatile *acc, OPER const &oper) const
  {
    // asserts in RAJA::util::reinterp_T_as_u and RAJA::util::reinterp_u_as_T
    // will enforce 64-bit T
    unsigned long long oldval, newval, readback;
    oldval = RAJA::util::reinterp_A_as_B<T, unsigned long long>(*acc);
    newval = RAJA::util::reinterp_A_as_B<T, unsigned long long>(
        oper(RAJA::util::reinterp_A_as_B<unsigned long long, T>(oldval)));
    while (
        (readback = cuda_atomic_CAS((unsigned long long volatile*)acc, oldval, newval)) !=
        oldval) {
      oldval = readback;
      newval = RAJA::util::reinterp_A_as_B<T, unsigned long long>(
          oper(RAJA::util::reinterp_A_as_B<unsigned long long, T>(oldval)));
    }
    return RAJA::util::reinterp_A_as_B<unsigned long long, T>(oldval);
  }
};


/*!
 * Generic impementation of any atomic 32-bit or 64-bit operator that can be
 * implemented using a compare and swap primitive.
 * Implementation uses the existing CUDA supplied unsigned 32-bit and 64-bit
 * CAS operators.
 * Returns the OLD value that was replaced by the result of this operation.
 */
template <typename T, typename OPER>
RAJA_INLINE __device__ T cuda_atomic_CAS_oper(T volatile *acc, OPER &&oper)
{
  CudaAtomicCAS<sizeof(T)> cas;
  return cas(acc, std::forward<OPER>(oper));
}


/*!
 * Catch-all policy passes off to CUDA's builtin atomics.
 *
 * This catch-all will only work for types supported by the compiler.
 * Specialization below can adapt for some unsupported types.
 *
 * These are atomic in cuda device code and non-atomic otherwise
 */
template <typename T>
RAJA_INLINE __device__ T cuda_atomicAdd(T volatile *acc, T value)
{
  return cuda_atomic_CAS_oper(acc, [=] __device__(T a) {
    return a + value;
  });
}

// 32-bit signed atomicAdd support by CUDA
template <>
RAJA_INLINE __device__ int cuda_atomicAdd<int>(int volatile *acc,
                                          int value)
{
  return ::atomicAdd((int *)acc, value);
}


// 32-bit unsigned atomicAdd support by CUDA
template <>
RAJA_INLINE __device__ unsigned cuda_atomicAdd<unsigned>(unsigned volatile *acc,
                                                    unsigned value)
{
  return ::atomicAdd((unsigned *)acc, value);
}

// 64-bit unsigned atomicAdd support by CUDA
template <>
RAJA_INLINE __device__ unsigned long long cuda_atomicAdd<unsigned long long>(
    unsigned long long volatile *acc,
    unsigned long long value)
{
  return ::atomicAdd((unsigned long long *)acc, value);
}


// 32-bit float atomicAdd support by CUDA
template <>
RAJA_INLINE __device__ float cuda_atomicAdd<float>(float volatile *acc,
                                              float value)
{
  return ::atomicAdd((float *)acc, value);
}


// 64-bit double atomicAdd support added for sm_60
#if __CUDA_ARCH__ >= 600
template <>
RAJA_INLINE __device__ double cuda_atomicAdd<double>(double volatile *acc,
                                                double value)
{
  return ::atomicAdd((double *)acc, value);
}
#endif

template <typename T>
RAJA_INLINE __device__ T cuda_atomicSub(T volatile *acc, T value)
{
  return cuda_atomic_CAS_oper(acc, [=] __device__(T a) {
    return a - value;
  });
}

// 32-bit signed atomicSub support by CUDA
template <>
RAJA_INLINE __device__ int cuda_atomicSub<int>(int volatile *acc,
                                          int value)
{
  return ::atomicSub((int *)acc, value);
}


// 32-bit unsigned atomicSub support by CUDA
template <>
RAJA_INLINE __device__ unsigned cuda_atomicSub<unsigned>(unsigned volatile *acc,
                                                    unsigned value)
{
  return ::atomicSub((unsigned *)acc, value);
}

template <typename T>
RAJA_INLINE __device__ T cuda_atomicMin(T volatile *acc, T value)
{
  return cuda_atomic_CAS_oper(acc, [=] __device__(T a) {
    return value < a ? value : a;
  });
}

// 32-bit signed atomicMin support by CUDA
template <>
RAJA_INLINE __device__ int cuda_atomicMin<int>(int volatile *acc,
                                          int value)
{
  return ::atomicMin((int *)acc, value);
}


// 32-bit unsigned atomicMin support by CUDA
template <>
RAJA_INLINE __device__ unsigned cuda_atomicMin<unsigned>(unsigned volatile *acc,
                                                    unsigned value)
{
  return ::atomicMin((unsigned *)acc, value);
}

// 64-bit unsigned atomicMin support by CUDA sm_35 and later
#if __CUDA_ARCH__ >= 350
template <>
RAJA_INLINE __device__ unsigned long long cuda_atomicMin<unsigned long long>(
    unsigned long long volatile *acc,
    unsigned long long value)
{
  return ::atomicMin((unsigned long long *)acc, value);
}
#endif

template <typename T>
RAJA_INLINE __device__ T cuda_atomicMax(T volatile *acc, T value)
{
  return cuda_atomic_CAS_oper(acc, [=] __device__(T a) {
    return value > a ? value : a;
  });
}

// 32-bit signed atomicMax support by CUDA
template <>
RAJA_INLINE __device__ int cuda_atomicMax<int>(int volatile *acc,
                                          int value)
{
  return ::atomicMax((int *)acc, value);
}


// 32-bit unsigned atomicMax support by CUDA
template <>
RAJA_INLINE __device__ unsigned cuda_atomicMax<unsigned>(unsigned volatile *acc,
                                                    unsigned value)
{
  return ::atomicMax((unsigned *)acc, value);
}

// 64-bit unsigned atomicMax support by CUDA sm_35 and later
#if __CUDA_ARCH__ >= 350
template <>
RAJA_INLINE __device__ unsigned long long cuda_atomicMax<unsigned long long>(
    unsigned long long volatile *acc,
    unsigned long long value)
{
  return ::atomicMax((unsigned long long *)acc, value);
}
#endif

template <typename T>
RAJA_INLINE __device__ T cuda_atomicInc(T volatile *acc, T val)
{
  // See:
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicinc
  return cuda_atomic_CAS_oper(acc, [=] __device__(T old) {
    return ((old >= val) ? 0 : (old + 1));
  });
}

// 32-bit unsigned atomicInc support by CUDA
template <>
RAJA_INLINE __device__ unsigned cuda_atomicInc<unsigned>(unsigned volatile *acc,
                                                    unsigned value)
{
  return ::atomicInc((unsigned *)acc, value);
}

template <typename T>
RAJA_INLINE __device__ T cuda_atomicInc(T volatile *acc)
{
  return cuda_atomic_CAS_oper(acc,
                                      [=] __device__(T a) { return a + 1; });
}


template <typename T>
RAJA_INLINE __device__ T cuda_atomicDec(T volatile *acc, T val)
{
  // See:
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicdec
  return cuda_atomic_CAS_oper(acc, [=] __device__(T old) {
    return (((old == 0) | (old > val)) ? val : (old - 1));
  });
}

// 32-bit unsigned atomicDec support by CUDA
template <>
RAJA_INLINE __device__ unsigned cuda_atomicDec<unsigned>(unsigned volatile *acc,
                                                    unsigned value)
{
  return ::atomicDec((unsigned *)acc, value);
}

template <typename T>
RAJA_INLINE __device__ T cuda_atomicDec(T volatile *acc)
{
  return cuda_atomic_CAS_oper(acc,
                                      [=] __device__(T a) { return a - 1; });
}


template <typename T>
RAJA_INLINE __device__ T cuda_atomicAnd(T volatile *acc, T value)
{
  return cuda_atomic_CAS_oper(acc, [=] __device__(T a) {
    return a & value;
  });
}

// 32-bit signed atomicAnd support by CUDA
template <>
RAJA_INLINE __device__ int cuda_atomicAnd<int>(int volatile *acc,
                                          int value)
{
  return ::atomicAnd((int *)acc, value);
}


// 32-bit unsigned atomicAnd support by CUDA
template <>
RAJA_INLINE __device__ unsigned cuda_atomicAnd<unsigned>(unsigned volatile *acc,
                                                    unsigned value)
{
  return ::atomicAnd((unsigned *)acc, value);
}

// 64-bit unsigned atomicAnd support by CUDA sm_35 and later
#if __CUDA_ARCH__ >= 350
template <>
RAJA_INLINE __device__ unsigned long long cuda_atomicAnd<unsigned long long>(
    unsigned long long volatile *acc,
    unsigned long long value)
{
  return ::atomicAnd((unsigned long long *)acc, value);
}
#endif

template <typename T>
RAJA_INLINE __device__ T cuda_atomicOr(T volatile *acc, T value)
{
  return cuda_atomic_CAS_oper(acc, [=] __device__(T a) {
    return a | value;
  });
}

// 32-bit signed atomicOr support by CUDA
template <>
RAJA_INLINE __device__ int cuda_atomicOr<int>(int volatile *acc,
                                         int value)
{
  return ::atomicOr((int *)acc, value);
}


// 32-bit unsigned atomicOr support by CUDA
template <>
RAJA_INLINE __device__ unsigned cuda_atomicOr<unsigned>(unsigned volatile *acc,
                                                   unsigned value)
{
  return ::atomicOr((unsigned *)acc, value);
}

// 64-bit unsigned atomicOr support by CUDA sm_35 and later
#if __CUDA_ARCH__ >= 350
template <>
RAJA_INLINE __device__ unsigned long long cuda_atomicOr<unsigned long long>(
    unsigned long long volatile *acc,
    unsigned long long value)
{
  return ::atomicOr((unsigned long long *)acc, value);
}
#endif

template <typename T>
RAJA_INLINE __device__ T cuda_atomicXor(T volatile *acc, T value)
{
  return cuda_atomic_CAS_oper(acc, [=] __device__(T a) {
    return a ^ value;
  });
}

// 32-bit signed atomicXor support by CUDA
template <>
RAJA_INLINE __device__ int cuda_atomicXor<int>(int volatile *acc,
                                          int value)
{
  return ::atomicXor((int *)acc, value);
}


// 32-bit unsigned atomicXor support by CUDA
template <>
RAJA_INLINE __device__ unsigned cuda_atomicXor<unsigned>(unsigned volatile *acc,
                                                    unsigned value)
{
  return ::atomicXor((unsigned *)acc, value);
}

// 64-bit unsigned atomicXor support by CUDA sm_35 and later
#if __CUDA_ARCH__ >= 350
template <>
RAJA_INLINE __device__ unsigned long long cuda_atomicXor<unsigned long long>(
    unsigned long long volatile *acc,
    unsigned long long value)
{
  return ::atomicXor((unsigned long long *)acc, value);
}
#endif

template <typename T>
RAJA_INLINE __device__ T cuda_atomicExchange(T volatile *acc, T value)
{
  // attempt to use the CUDA builtin atomic, if it exists for T
  return ::atomicExch((T *)acc, value);
}


template <typename T>
RAJA_INLINE __device__ T cuda_atomicCAS(T volatile *acc, T compare, T value)
{
  return cuda_atomic_CAS(acc, compare, value);
}

template <>
RAJA_INLINE __device__ int cuda_atomicCAS<int>(
    int volatile *acc, int compare, int value)
{
  return ::atomicCAS((int *)acc, compare, value);
}

template <>
RAJA_INLINE __device__ unsigned cuda_atomicCAS<unsigned>(
    unsigned volatile *acc, unsigned compare, unsigned value)
{
  return ::atomicCAS((unsigned *)acc, compare, value);
}

template <>
RAJA_INLINE __device__ unsigned long long cuda_atomicCAS<unsigned long long>(
    unsigned long long volatile *acc,
    unsigned long long compare,
    unsigned long long value)
{
  return ::atomicCAS((unsigned long long *)acc, compare, value);
}

}  // namespace detail


struct cuda_atomic {
};


/*!
 * Catch-all policy passes off to CUDA's builtin atomics.
 *
 * This catch-all will only work for types supported by the compiler.
 * Specialization below can adapt for some unsupported types.
 *
 * These are atomic in cuda device code and non-atomic otherwise
 */
RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicAdd(cuda_atomic, T volatile *acc, T value)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicAdd(acc, value);
#else
  T old = *acc;
  *acc = old + value;
  return old;
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicSub(cuda_atomic, T volatile *acc, T value)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicSub(acc, value);
#else
  T old = *acc;
  *acc = old - value;
  return old;
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicMin(cuda_atomic, T volatile *acc, T value)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicMin(acc, value);
#else
  T old = *acc;
  if (value < old) *acc = value;
  return old;
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicMax(cuda_atomic, T volatile *acc, T value)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicMax(acc, value);
#else
  T old = *acc;
  if (value > old) *acc = value;
  return old;
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicInc(cuda_atomic, T volatile *acc, T val)
{
#ifdef __CUDA_ARCH__
  // See:
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicinc
  return detail::cuda_atomicInc(acc, val);
#else
  T old = *acc;
  *acc = ((old >= val) ? 0 : (old + 1));
  return old;
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicInc(cuda_atomic, T volatile *acc)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicInc(acc);
#else
  T old = *acc;
  *acc = old + 1;
  return old;
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicDec(cuda_atomic, T volatile *acc, T val)
{
#ifdef __CUDA_ARCH__
  // See:
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicdec
  return detail::cuda_atomicDec(acc, val);
#else
  T old = *acc;
  *acc = (((old == 0) | (old > val)) ? val : (old - 1));
  return old;
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicDec(cuda_atomic, T volatile *acc)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicDec(acc);
#else
  T old = *acc;
  *acc = old - 1;
  return old;
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicAnd(cuda_atomic, T volatile *acc, T value)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicAnd(acc, value);
#else
  T old = *acc;
  *acc = old & value;
  return old;
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicOr(cuda_atomic, T volatile *acc, T value)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicOr(acc, value);
#else
  T old = *acc;
  *acc = old | value;
  return old;
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE RAJA_HOST_DEVICE T atomicXor(cuda_atomic, T volatile *acc, T value)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicXor(acc, value);
#else
  T old = *acc;
  *acc = old ^ value;
  return old;
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicExchange(cuda_atomic, T volatile *acc, T value)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicExchange(acc, value);
#else
  T old = *acc;
  *acc = value;
  return old;
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicCAS(cuda_atomic, T volatile *acc, T compare, T value)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicCAS(acc, compare, value);
#else
  T old = *acc;
  if (old == compare) *acc = value;
  return old;
#endif
}

}  // namespace atomic
}  // namespace RAJA


#endif  // RAJA_ENABLE_CUDA
#endif  // guard
