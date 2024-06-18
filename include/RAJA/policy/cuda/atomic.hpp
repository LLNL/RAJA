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
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_cuda_atomic_HPP
#define RAJA_policy_cuda_atomic_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include <stdexcept>
#include <type_traits>

#if __CUDA__ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 6
#include <cuda/atomic>
#endif

#include "RAJA/policy/sequential/atomic.hpp"
#include "RAJA/policy/atomic_builtin.hpp"
#if defined(RAJA_ENABLE_OPENMP)
#include "RAJA/policy/openmp/atomic.hpp"
#endif

#include "RAJA/util/Operators.hpp"
#include "RAJA/util/TypeConvert.hpp"
#include "RAJA/util/macros.hpp"


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


/*!
 * Atomic exchange
 */
template <typename T,
          std::enable_if_t<std::is_same<T, int>::value ||
                           std::is_same<T, unsigned int>::value ||
                           std::is_same<T, unsigned long long int>::value ||
                           std::is_same<T, float>::value, bool> = true>
RAJA_INLINE __device__ T cuda_atomicExchange(T *acc, T value)
{
  return ::atomicExch(acc, value);
}

template <typename T,
          std::enable_if_t<!std::is_same<T, int>::value &&
                           !std::is_same<T, unsigned int>::value &&
                           !std::is_same<T, unsigned long long int>::value &&
                           !std::is_same<T, float>::value &&
                           sizeof(T) == sizeof(unsigned int), bool> = true>
RAJA_INLINE __device__ T cuda_atomicExchange(T *acc, T value)
{
  return RAJA::util::reinterp_A_as_B<unsigned int, T>(
    cuda_atomicExchange(reinterpret_cast<unsigned int*>(acc),
                        reinterpret_cast<unsigned int&>(value)));
}

template <typename T,
          std::enable_if_t<!std::is_same<T, int>::value &&
                           !std::is_same<T, unsigned int>::value &&
                           !std::is_same<T, unsigned long long int>::value &&
                           !std::is_same<T, float>::value &&
                           sizeof(T) == sizeof(unsigned long long int), bool> = true>
RAJA_INLINE __device__ T cuda_atomicExchange(T *acc, T value)
{
  return RAJA::util::reinterp_A_as_B<unsigned long long int, T>(
    cuda_atomicExchange(reinterpret_cast<unsigned long long int*>(acc),
                        reinterpret_cast<unsigned long long int&>(value)));
}


/*!
 * Atomic load and store
 */
#if __CUDA__ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 6

template <typename T>
RAJA_INLINE __device__ T cuda_atomicLoad(T *acc)
{
  return cuda::atomic_ref<T, cuda::thread_scope_device>(*acc).load(
    cuda::memory_order_relaxed{});
}


template <typename T>
RAJA_INLINE __device__ void cuda_atomicStore(T *acc, T value)
{
  cuda::atomic_ref<T, cuda::thread_scope_device>(*acc).store(
    value, cuda::memory_order_relaxed{});
}

#else

template <typename T,
          std::enable_if_t<std::is_same<T, int>::value ||
                           std::is_same<T, unsigned int>::value ||
                           std::is_same<T, unsigned long long int>::value, bool> = true>
RAJA_INLINE __device__ T cuda_atomicLoad(T *acc)
{
  return ::atomicOr(acc, static_cast<T>(0));
}

template <typename T,
          std::enable_if_t<!std::is_same<T, int>::value &&
                           !std::is_same<T, unsigned int>::value &&
                           !std::is_same<T, unsigned long long int>::value &&
                           sizeof(T) == sizeof(unsigned int), bool> = true>
RAJA_INLINE __device__ T cuda_atomicLoad(T *acc)
{
  return RAJA::util::reinterp_A_as_B<unsigned int, T>(
    cuda_atomicLoad(reinterpret_cast<unsigned int*>(acc)));
}

template <typename T,
          std::enable_if_t<!std::is_same<T, int>::value &&
                           !std::is_same<T, unsigned int>::value &&
                           !std::is_same<T, unsigned long long int>::value &&
                           sizeof(T) == sizeof(unsigned long long int), bool> = true>
RAJA_INLINE __device__ T cuda_atomicLoad(T *acc)
{
  return RAJA::util::reinterp_A_as_B<unsigned long long int, T>(
    cuda_atomicLoad(reinterpret_cast<unsigned long long int*>(acc)));
}


template <typename T>
RAJA_INLINE __device__ void cuda_atomicStore(T *acc, T value)
{
  cuda_atomicExchange(acc, value);
}

#endif


/*!
 * Atomic compare and swap
 */
template <typename T,
          std::enable_if_t<
#if __CUDA_ARCH__ >= 700
                           std::is_same<T, unsigned short int>::value ||
#endif
                           std::is_same<T, int>::value ||
                           std::is_same<T, unsigned int>::value ||
                           std::is_same<T, unsigned long long int>::value, bool> = true>
RAJA_INLINE __device__ T cuda_atomicCAS(T *acc, T compare, T value)
{
  return ::atomicCAS(acc, compare, value);
}

#if __CUDA_ARCH__ >= 700
template <typename T,
          std::enable_if_t<!std::is_same<T, unsigned short int>::value &&
                           !std::is_same<T, int>::value &&
                           !std::is_same<T, unsigned int>::value &&
                           !std::is_same<T, unsigned long long int>::value &&
                           sizeof(T) == sizeof(unsigned short int), bool> = true>
RAJA_INLINE __device__ T cuda_atomicCAS(T *acc, T compare, T value)
{
  return RAJA::util::reinterp_A_as_B<unsigned short int, T>(
    cuda_atomicCAS(reinterpret_cast<unsigned short int*>(acc),
                   reinterpret_cast<unsigned short int&>(compare),
                   reinterpret_cast<unsigned short int&>(value)));
}
#endif

template <typename T,
          std::enable_if_t<
#if __CUDA_ARCH__ >= 700
                           !std::is_same<T, unsigned short int>::value &&
#endif
                           !std::is_same<T, int>::value &&
                           !std::is_same<T, unsigned int>::value &&
                           !std::is_same<T, unsigned long long int>::value &&
                           sizeof(T) == sizeof(unsigned int), bool> = true>
RAJA_INLINE __device__ T cuda_atomicCAS(T *acc, T compare, T value)
{
  return RAJA::util::reinterp_A_as_B<unsigned int, T>(
    cuda_atomicCAS(reinterpret_cast<unsigned int*>(acc),
                   reinterpret_cast<unsigned int&>(compare),
                   reinterpret_cast<unsigned int&>(value)));
}

template <typename T,
          std::enable_if_t<
#if __CUDA_ARCH__ >= 700
                           !std::is_same<T, unsigned short int>::value &&
#endif
                           !std::is_same<T, int>::value &&
                           !std::is_same<T, unsigned int>::value &&
                           !std::is_same<T, unsigned long long int>::value &&
                           sizeof(T) == sizeof(unsigned long long int), bool> = true>
RAJA_INLINE __device__ T cuda_atomicCAS(T *acc, T compare, T value)
{
  return RAJA::util::reinterp_A_as_B<unsigned long long int, T>(
    cuda_atomicCAS(reinterpret_cast<unsigned long long int*>(acc),
                   reinterpret_cast<unsigned long long int&>(compare),
                   reinterpret_cast<unsigned long long int&>(value)));
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
RAJA_INLINE __device__ bool cuda_atomicCAS_equal(const T& a, const T& b)
{
  return a == b;
}

template <typename T,
          std::enable_if_t<!std::is_same<T, int>::value &&
                           !std::is_same<T, unsigned int>::value &&
                           !std::is_same<T, unsigned long long int>::value &&
                           sizeof(T) == sizeof(unsigned int), bool> = true>
RAJA_INLINE __device__ bool cuda_atomicCAS_equal(const T& a, const T& b)
{
  return reinterpret_cast<const unsigned int&>(a) ==
         reinterpret_cast<const unsigned int&>(b);
}

template <typename T,
          std::enable_if_t<!std::is_same<T, int>::value &&
                           !std::is_same<T, unsigned int>::value &&
                           !std::is_same<T, unsigned long long int>::value &&
                           sizeof(T) == sizeof(unsigned long long int), bool> = true>
RAJA_INLINE __device__ bool cuda_atomicCAS_equal(const T& a, const T& b)
{
  return reinterpret_cast<const unsigned long long int&>(a) ==
         reinterpret_cast<const unsigned long long int&>(b);
}


/*!
 * Generic impementation of any atomic 32-bit or 64-bit operator.
 * Implementation uses the existing CUDA supplied unsigned 32-bit or 64-bit CAS
 * operator. Returns the OLD value that was replaced by the result of this
 * operation.
 */
template <typename T, typename OPER>
RAJA_INLINE __device__ T cuda_atomicCAS(T *acc, OPER&& oper)
{
  T old = cuda_atomicLoad(acc);
  T expected;

  do {
    expected = old;
    old = cuda_atomicCAS(acc, expected, oper(expected));
  } while (!cuda_atomicCAS_equal(old, expected));

  return old;
}


/*!
 * Atomic addition
 */
using cuda_atomicAdd_builtin_types = list<
      int
     ,unsigned int
     ,unsigned long long int
     ,float
#if __CUDA_ARCH__ >= 600
     ,double
#endif
    >;

template <typename T,
          enable_if_is_none_of<T, cuda_atomicAdd_builtin_types>* = nullptr>
RAJA_INLINE __device__ T cuda_atomicAdd(T *acc, T value)
{
  return cuda_atomicCAS(acc, [value] __device__(T old) {
    return old + value;
  });
}

template <typename T,
          enable_if_is_any_of<T, cuda_atomicAdd_builtin_types>* = nullptr>
RAJA_INLINE __device__ T cuda_atomicAdd(T *acc, T value)
{
  return ::atomicAdd(acc, value);
}


/*!
 * Atomic subtract
 */
using cuda_atomicSub_builtin_types = cuda_atomicAdd_builtin_types;

using cuda_atomicSub_via_Sub_builtin_types = list<
      int
     ,unsigned int
    >;

using cuda_atomicSub_via_Add_builtin_types = list<
      unsigned long long int
     ,float
#if __CUDA_ARCH__ >= 600
     ,double
#endif
    >;

template <typename T,
          enable_if_is_none_of<T, cuda_atomicSub_builtin_types>* = nullptr>
RAJA_INLINE __device__ T cuda_atomicSub(T *acc, T value)
{
  return cuda_atomicCAS(acc, [value] __device__(T old) {
    return old - value;
  });
}

template <typename T,
          enable_if_is_any_of<T, cuda_atomicSub_via_Sub_builtin_types>* = nullptr>
RAJA_INLINE __device__ T cuda_atomicSub(T *acc, T value)
{
  return ::atomicSub(acc, value);
}

template <typename T,
          enable_if_is_any_of<T, cuda_atomicSub_via_Add_builtin_types>* = nullptr>
RAJA_INLINE __device__ T cuda_atomicSub(T *acc, T value)
{
  return ::atomicAdd(acc, -value);
}


/*!
 * Atomic min/max
 */
using cuda_atomicMinMax_builtin_types = list<
      int
     ,unsigned int
#if __CUDA_ARCH__ >= 500
     ,long long int
     ,unsigned long long int
#endif
    >;


/*!
 * Atomic min
 */
template <typename T,
          enable_if_is_none_of<T, cuda_atomicMinMax_builtin_types>* = nullptr>
RAJA_INLINE __device__ T cuda_atomicMin(T *acc, T value)
{
  return cuda_atomicCAS(acc, [value] __device__(T old) {
    return value < old ? value : old;
  });
}

template <typename T,
          enable_if_is_any_of<T, cuda_atomicMinMax_builtin_types>* = nullptr>
RAJA_INLINE __device__ T cuda_atomicMin(T *acc, T value)
{
  return ::atomicMin(acc, value);
}


/*!
 * Atomic max
 */
template <typename T,
          enable_if_is_none_of<T, cuda_atomicMinMax_builtin_types>* = nullptr>
RAJA_INLINE __device__ T cuda_atomicMax(T *acc, T value)
{
  return cuda_atomicCAS(acc, [value] __device__(T old) {
    return old < value ? value : old;
  });
}

template <typename T,
          enable_if_is_any_of<T, cuda_atomicMinMax_builtin_types>* = nullptr>
RAJA_INLINE __device__ T cuda_atomicMax(T *acc, T value)
{
  return ::atomicMax(acc, value);
}


/*!
 * Atomic increment/decrement with reset
 */
using cuda_atomicIncDecReset_builtin_types = list<
      unsigned int
    >;


/*!
 * Atomic increment with reset
 */
template <typename T,
          enable_if_is_none_of<T, cuda_atomicIncDecReset_builtin_types>* = nullptr>
RAJA_INLINE __device__ T cuda_atomicInc(T *acc, T value)
{
  // See:
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicinc
  return cuda_atomicCAS(acc, [value] __device__(T old) {
    return value <= old ? static_cast<T>(0) : old + static_cast<T>(1);
  });
}

template <typename T,
          enable_if_is_any_of<T, cuda_atomicIncDecReset_builtin_types>* = nullptr>
RAJA_INLINE __device__ T cuda_atomicInc(T *acc, T value)
{
  return ::atomicInc(acc, value);
}


/*!
 * Atomic increment (implemented in terms of atomic addition)
 */
template <typename T>
RAJA_INLINE __device__ T cuda_atomicInc(T *acc)
{
  return cuda_atomicAdd(acc, static_cast<T>(1));
}


/*!
 * Atomic decrement with reset
 */
template <typename T,
          enable_if_is_none_of<T, cuda_atomicIncDecReset_builtin_types>* = nullptr>
RAJA_INLINE __device__ T cuda_atomicDec(T *acc, T value)
{
  // See:
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicdec
  return cuda_atomicCAS(acc, [value] __device__(T old) {
    return old == static_cast<T>(0) || value < old ? value : old - static_cast<T>(1);
  });
}

template <typename T,
          enable_if_is_any_of<T, cuda_atomicIncDecReset_builtin_types>* = nullptr>
RAJA_INLINE __device__ T cuda_atomicDec(T *acc, T value)
{
  return ::atomicDec(acc, value);
}


/*!
 * Atomic decrement (implemented in terms of atomic subtraction)
 */
template <typename T>
RAJA_INLINE __device__ T cuda_atomicDec(T *acc)
{
  return cuda_atomicSub(acc, static_cast<T>(1));
}


/*!
 * Atomic bitwise functions (and, or, xor)
 */
using cuda_atomicBit_builtin_types = list<
      int
     ,unsigned int
     ,unsigned long long int
    >;


/*!
 * Atomic and
 */
template <typename T,
          enable_if_is_none_of<T, cuda_atomicBit_builtin_types>* = nullptr>
RAJA_INLINE __device__ T cuda_atomicAnd(T *acc, T value)
{
  return cuda_atomicCAS(acc, [value] __device__(T old) {
    return old & value;
  });
}

template <typename T,
          enable_if_is_any_of<T, cuda_atomicBit_builtin_types>* = nullptr>
RAJA_INLINE __device__ T cuda_atomicAnd(T *acc, T value)
{
  return ::atomicAnd(acc, value);
}


/*!
 * Atomic or
 */
template <typename T,
          enable_if_is_none_of<T, cuda_atomicBit_builtin_types>* = nullptr>
RAJA_INLINE __device__ T cuda_atomicOr(T *acc, T value)
{
  return cuda_atomicCAS(acc, [value] __device__(T old) {
    return old | value;
  });
}

template <typename T,
          enable_if_is_any_of<T, cuda_atomicBit_builtin_types>* = nullptr>
RAJA_INLINE __device__ T cuda_atomicOr(T *acc, T value)
{
  return ::atomicOr(acc, value);
}


/*!
 * Atomic xor
 */
template <typename T,
          enable_if_is_none_of<T, cuda_atomicBit_builtin_types>* = nullptr>
RAJA_INLINE __device__ T cuda_atomicXor(T *acc, T value)
{
  return cuda_atomicCAS(acc, [value] __device__(T old) {
    return old ^ value;
  });
}

template <typename T,
          enable_if_is_any_of<T, cuda_atomicBit_builtin_types>* = nullptr>
RAJA_INLINE __device__ T cuda_atomicXor(T *acc, T value)
{
  return ::atomicXor(acc, value);
}


}  // namespace detail


/*!
 * Catch-all policy passes off to CUDA's builtin atomics.
 *
 * This catch-all will only work for types supported by the compiler.
 * Specialization below can adapt for some unsupported types.
 *
 * These are atomic in cuda device code and non-atomic otherwise
 */
RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicLoad(cuda_atomic_explicit<host_policy>, T *acc)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicLoad(acc);
#else
  return RAJA::atomicLoad(host_policy{}, acc);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE void
atomicStore(cuda_atomic_explicit<host_policy>, T *acc, T value)
{
#ifdef __CUDA_ARCH__
  detail::cuda_atomicStore(acc, value);
#else
  RAJA::atomicStore(host_policy{}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicAdd(cuda_atomic_explicit<host_policy>, T *acc, T value)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicAdd(acc, value);
#else
  return RAJA::atomicAdd(host_policy{}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicSub(cuda_atomic_explicit<host_policy>, T *acc, T value)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicSub(acc, value);
#else
  return RAJA::atomicSub(host_policy{}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicMin(cuda_atomic_explicit<host_policy>, T *acc, T value)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicMin(acc, value);
#else
  return RAJA::atomicMin(host_policy{}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicMax(cuda_atomic_explicit<host_policy>, T *acc, T value)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicMax(acc, value);
#else
  return RAJA::atomicMax(host_policy{}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicInc(cuda_atomic_explicit<host_policy>, T *acc, T value)
{
#ifdef __CUDA_ARCH__
  // See:
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicinc
  return detail::cuda_atomicInc(acc, value);
#else
  return RAJA::atomicInc(host_policy{}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicInc(cuda_atomic_explicit<host_policy>, T *acc)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicInc(acc);
#else
  return RAJA::atomicInc(host_policy{}, acc);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicDec(cuda_atomic_explicit<host_policy>, T *acc, T value)
{
#ifdef __CUDA_ARCH__
  // See:
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicdec
  return detail::cuda_atomicDec(acc, value);
#else
  return RAJA::atomicDec(host_policy{}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicDec(cuda_atomic_explicit<host_policy>, T *acc)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicDec(acc);
#else
  return RAJA::atomicDec(host_policy{}, acc);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicAnd(cuda_atomic_explicit<host_policy>, T *acc, T value)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicAnd(acc, value);
#else
  return RAJA::atomicAnd(host_policy{}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicOr(cuda_atomic_explicit<host_policy>, T *acc, T value)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicOr(acc, value);
#else
  return RAJA::atomicOr(host_policy{}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicXor(cuda_atomic_explicit<host_policy>, T *acc, T value)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicXor(acc, value);
#else
  return RAJA::atomicXor(host_policy{}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicExchange(cuda_atomic_explicit<host_policy>, T *acc, T value)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicExchange(acc, value);
#else
  return RAJA::atomicExchange(host_policy{}, acc, value);
#endif
}

RAJA_SUPPRESS_HD_WARN
template <typename T, typename host_policy>
RAJA_INLINE RAJA_HOST_DEVICE T
atomicCAS(cuda_atomic_explicit<host_policy>, T *acc, T compare, T value)
{
#ifdef __CUDA_ARCH__
  return detail::cuda_atomicCAS(acc, compare, value);
#else
  return RAJA::atomicCAS(host_policy{}, acc, compare, value);
#endif
}

}  // namespace RAJA


#endif  // RAJA_ENABLE_CUDA
#endif  // guard
