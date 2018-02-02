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
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
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
#include "RAJA/util/Operators.hpp"
#include "RAJA/util/TypeConvert.hpp"
#include "RAJA/util/defines.hpp"

#include <stdexcept>

#if defined(RAJA_ENABLE_CUDA)


namespace RAJA
{
namespace atomic
{


namespace detail
{

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
    while ((readback = ::atomicCAS((unsigned *)acc, oldval, newval))
           != oldval) {
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
    while ((readback = ::atomicCAS((unsigned long long *)acc, oldval, newval))
           != oldval) {
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


}  // namespace detail


struct cuda_atomic {
};


/*!
 * Catch-all policy passes off to CUDA's builtin atomics.
 *
 * This catch-all will only work for types supported by the compiler.
 * Specialization below can adapt for some unsupported types.
 */
template <typename T>
RAJA_INLINE __device__ T atomicAdd(cuda_atomic, T volatile *acc, T value)
{
  return detail::cuda_atomic_CAS_oper(acc, [=] __device__(T a) {
    return a + value;
  });
}


// 32-bit signed atomicAdd support by CUDA
template <>
RAJA_INLINE __device__ int atomicAdd<int>(cuda_atomic,
                                          int volatile *acc,
                                          int value)
{
  return ::atomicAdd((int *)acc, value);
}


// 32-bit unsigned atomicAdd support by CUDA
template <>
RAJA_INLINE __device__ unsigned atomicAdd<unsigned>(cuda_atomic,
                                                    unsigned volatile *acc,
                                                    unsigned value)
{
  return ::atomicAdd((unsigned *)acc, value);
}

// 64-bit unsigned atomicAdd support by CUDA
template <>
RAJA_INLINE __device__ unsigned long long atomicAdd<unsigned long long>(
    cuda_atomic,
    unsigned long long volatile *acc,
    unsigned long long value)
{
  return ::atomicAdd((unsigned long long *)acc, value);
}


// 32-bit float atomicAdd support by CUDA
template <>
RAJA_INLINE __device__ float atomicAdd<float>(cuda_atomic,
                                              float volatile *acc,
                                              float value)
{
  return ::atomicAdd((float *)acc, value);
}


// 64-bit double atomicAdd support added for sm_60
#if __CUDA_ARCH__ >= 600
template <>
RAJA_INLINE __device__ double atomicAdd<double>(cuda_atomic,
                                                double volatile *acc,
                                                double value)
{
  return ::atomicAdd((double *)acc, value);
}
#endif


template <typename T>
RAJA_INLINE __device__ T atomicSub(cuda_atomic, T volatile *acc, T value)
{
  return detail::cuda_atomic_CAS_oper(acc, [=] __device__(T a) {
    return a - value;
  });
}

// 32-bit signed atomicSub support by CUDA
template <>
RAJA_INLINE __device__ int atomicSub<int>(cuda_atomic,
                                          int volatile *acc,
                                          int value)
{
  return ::atomicSub((int *)acc, value);
}


// 32-bit unsigned atomicSub support by CUDA
template <>
RAJA_INLINE __device__ unsigned atomicSub<unsigned>(cuda_atomic,
                                                    unsigned volatile *acc,
                                                    unsigned value)
{
  return ::atomicSub((unsigned *)acc, value);
}


template <typename T>
RAJA_INLINE __device__ T atomicMin(cuda_atomic, T volatile *acc, T value)
{
  return detail::cuda_atomic_CAS_oper(acc, [=] __device__(T a) {
    return a < value ? a : value;
  });
}


// 32-bit signed atomicMin support by CUDA
template <>
RAJA_INLINE __device__ int atomicMin<int>(cuda_atomic,
                                          int volatile *acc,
                                          int value)
{
  return ::atomicMin((int *)acc, value);
}


// 32-bit unsigned atomicMin support by CUDA
template <>
RAJA_INLINE __device__ unsigned atomicMin<unsigned>(cuda_atomic,
                                                    unsigned volatile *acc,
                                                    unsigned value)
{
  return ::atomicMin((unsigned *)acc, value);
}

// 64-bit unsigned atomicMin support by CUDA sm_35 and later
#if __CUDA_ARCH__ >= 350
template <>
RAJA_INLINE __device__ unsigned long long atomicMin<unsigned long long>(
    cuda_atomic,
    unsigned long long volatile *acc,
    unsigned long long value)
{
  return ::atomicMin((unsigned long long *)acc, value);
}
#endif


template <typename T>
RAJA_INLINE __device__ T atomicMax(cuda_atomic, T volatile *acc, T value)
{
  return detail::cuda_atomic_CAS_oper(acc, [=] __device__(T a) {
    return a > value ? a : value;
  });
}

// 32-bit signed atomicMax support by CUDA
template <>
RAJA_INLINE __device__ int atomicMax<int>(cuda_atomic,
                                          int volatile *acc,
                                          int value)
{
  return ::atomicMax((int *)acc, value);
}


// 32-bit unsigned atomicMax support by CUDA
template <>
RAJA_INLINE __device__ unsigned atomicMax<unsigned>(cuda_atomic,
                                                    unsigned volatile *acc,
                                                    unsigned value)
{
  return ::atomicMax((unsigned *)acc, value);
}

// 64-bit unsigned atomicMax support by CUDA sm_35 and later
#if __CUDA_ARCH__ >= 350
template <>
RAJA_INLINE __device__ unsigned long long atomicMax<unsigned long long>(
    cuda_atomic,
    unsigned long long volatile *acc,
    unsigned long long value)
{
  return ::atomicMax((unsigned long long *)acc, value);
}
#endif


template <typename T>
RAJA_INLINE __device__ T atomicInc(cuda_atomic, T volatile *acc, T val)
{
  // See:
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicinc
  return detail::cuda_atomic_CAS_oper(acc, [=] __device__(T old) {
    return ((old >= val) ? 0 : (old + 1));
  });
}

// 32-bit unsigned atomicInc support by CUDA
template <>
RAJA_INLINE __device__ unsigned atomicInc<unsigned>(cuda_atomic,
                                                    unsigned volatile *acc,
                                                    unsigned value)
{
  return ::atomicInc((unsigned *)acc, value);
}


template <typename T>
RAJA_INLINE __device__ T atomicInc(cuda_atomic, T volatile *acc)
{
  return detail::cuda_atomic_CAS_oper(acc,
                                      [=] __device__(T a) { return a + 1; });
}


template <typename T>
RAJA_INLINE __device__ T atomicDec(cuda_atomic, T volatile *acc, T val)
{
  // See:
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicdec
  return detail::cuda_atomic_CAS_oper(acc, [=] __device__(T old) {
    return (((old == 0) | (old > val)) ? val : (old - 1));
  });
}


// 32-bit unsigned atomicDec support by CUDA
template <>
RAJA_INLINE __device__ unsigned atomicDec<unsigned>(cuda_atomic,
                                                    unsigned volatile *acc,
                                                    unsigned value)
{
  return ::atomicDec((unsigned *)acc, value);
}


template <typename T>
RAJA_INLINE __device__ T atomicDec(cuda_atomic, T volatile *acc)
{
  return detail::cuda_atomic_CAS_oper(acc,
                                      [=] __device__(T a) { return a - 1; });
}


template <typename T>
RAJA_INLINE __device__ T atomicAnd(cuda_atomic, T volatile *acc, T value)
{
  return detail::cuda_atomic_CAS_oper(acc, [=] __device__(T a) {
    return a & value;
  });
}

// 32-bit signed atomicAnd support by CUDA
template <>
RAJA_INLINE __device__ int atomicAnd<int>(cuda_atomic,
                                          int volatile *acc,
                                          int value)
{
  return ::atomicAnd((int *)acc, value);
}


// 32-bit unsigned atomicAnd support by CUDA
template <>
RAJA_INLINE __device__ unsigned atomicAnd<unsigned>(cuda_atomic,
                                                    unsigned volatile *acc,
                                                    unsigned value)
{
  return ::atomicAnd((unsigned *)acc, value);
}

// 64-bit unsigned atomicAnd support by CUDA sm_35 and later
#if __CUDA_ARCH__ >= 350
template <>
RAJA_INLINE __device__ unsigned long long atomicAnd<unsigned long long>(
    cuda_atomic,
    unsigned long long volatile *acc,
    unsigned long long value)
{
  return ::atomicAnd((unsigned long long *)acc, value);
}
#endif


template <typename T>
RAJA_INLINE __device__ T atomicOr(cuda_atomic, T volatile *acc, T value)
{
  return detail::cuda_atomic_CAS_oper(acc, [=] __device__(T a) {
    return a | value;
  });
}

// 32-bit signed atomicOr support by CUDA
template <>
RAJA_INLINE __device__ int atomicOr<int>(cuda_atomic,
                                         int volatile *acc,
                                         int value)
{
  return ::atomicOr((int *)acc, value);
}


// 32-bit unsigned atomicOr support by CUDA
template <>
RAJA_INLINE __device__ unsigned atomicOr<unsigned>(cuda_atomic,
                                                   unsigned volatile *acc,
                                                   unsigned value)
{
  return ::atomicOr((unsigned *)acc, value);
}

// 64-bit unsigned atomicOr support by CUDA sm_35 and later
#if __CUDA_ARCH__ >= 350
template <>
RAJA_INLINE __device__ unsigned long long atomicOr<unsigned long long>(
    cuda_atomic,
    unsigned long long volatile *acc,
    unsigned long long value)
{
  return ::atomicOr((unsigned long long *)acc, value);
}
#endif


template <typename T>
RAJA_INLINE __device__ T atomicXor(cuda_atomic, T volatile *acc, T value)
{
  return detail::cuda_atomic_CAS_oper(acc, [=] __device__(T a) {
    return a ^ value;
  });
}

// 32-bit signed atomicXor support by CUDA
template <>
RAJA_INLINE __device__ int atomicXor<int>(cuda_atomic,
                                          int volatile *acc,
                                          int value)
{
  return ::atomicXor((int *)acc, value);
}


// 32-bit unsigned atomicXor support by CUDA
template <>
RAJA_INLINE __device__ unsigned atomicXor<unsigned>(cuda_atomic,
                                                    unsigned volatile *acc,
                                                    unsigned value)
{
  return ::atomicXor((unsigned *)acc, value);
}

// 64-bit unsigned atomicXor support by CUDA sm_35 and later
#if __CUDA_ARCH__ >= 350
template <>
RAJA_INLINE __device__ unsigned long long atomicXor<unsigned long long>(
    cuda_atomic,
    unsigned long long volatile *acc,
    unsigned long long value)
{
  return ::atomicXor((unsigned long long *)acc, value);
}
#endif


template <typename T>
RAJA_INLINE __device__ T atomicExchange(cuda_atomic, T volatile *acc, T value)
{
  // attempt to use the CUDA builtin atomic, if it exists for T
  return ::atomicExch((T *)acc, value);
}


template <typename T>
RAJA_INLINE __device__ T
atomicCAS(cuda_atomic, T volatile *acc, T compare, T value)
{
  // attempt to use the CUDA builtin atomic, if it exists for T
  return ::atomicCAS((T *)acc, compare, value);
}


}  // namespace atomic
}  // namespace RAJA


#endif  // RAJA_ENABLE_CUDA
#endif  // guard
