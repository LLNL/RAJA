/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA intrinsics templates for CUDA execution.
 *
 *          These methods should work on any platform that supports
 *          CUDA devices.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_cuda_intrinsics_HPP
#define RAJA_cuda_intrinsics_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include <type_traits>

#include <cuda.h>

#include "RAJA/util/macros.hpp"
#include "RAJA/util/SoAArray.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/cuda/policy.hpp"


namespace RAJA
{

namespace cuda
{

namespace impl
{

/*!
 * \brief Abstracts access to memory when coordinating between threads at
 *       device scope. The fences provided here are to be used with relaxed
 *       atomics in order to guarantee memory ordering and visibility of the
 *       accesses done through this class.
 *
 * \Note This uses device scope fences to ensure ordering and to flush local
 *       caches so that memory accesses become visible to the whole device.
 * \Note This class uses normal memory accesses that are cached in local caches
 *       so device scope fences are required to make memory accesses visible
 *       to the whole device.
 */
struct AccessorDeviceScopeUseDeviceFence : RAJA::detail::DefaultAccessor
{
  static RAJA_DEVICE RAJA_INLINE void fence_acquire()
  {
    __threadfence();
  }

  static RAJA_DEVICE RAJA_INLINE void fence_release()
  {
    __threadfence();
  }
};

/*!
 ******************************************************************************
 *
 * \brief Abstracts access to memory when coordinating between threads at
 *       device scope. The fences provided here are to be used with relaxed
 *       atomics in order to guarantee memory ordering and visibility of the
 *       accesses done through this class.
 *
 * \Note This may use block scope fences to ensure ordering and avoid flushing
 *       local caches so special memory accesses are used to ensure visibility
 *       to the whole device.
 * \Note This class uses device scope atomic memory accesses to bypass local
 *       caches so memory accesses are visible to the whole device without
 *       device scope fences.
 * \Note A memory access may be split into multiple memory accesses, so
 *       even though atomic instructions are used concurrent accesses between
 *       different threads are not thread safe.
 *
 ******************************************************************************
 */
struct AccessorDeviceScopeUseBlockFence
{
  // cuda has 32 and 64 bit atomics
  static constexpr size_t min_atomic_int_type_size = sizeof(unsigned int);
  static constexpr size_t max_atomic_int_type_size = sizeof(unsigned long long);

  template < typename T >
  static RAJA_DEVICE RAJA_INLINE T get(T* in_ptr, size_t idx)
  {
    using ArrayType = RAJA::detail::AsIntegerArray<T, min_atomic_int_type_size, max_atomic_int_type_size>;
    using integer_type = typename ArrayType::integer_type;

    ArrayType u;
    auto ptr = const_cast<integer_type*>(reinterpret_cast<const integer_type*>(in_ptr + idx));

    for (size_t i = 0; i < u.array_size(); ++i) {
      u.array[i] = atomicAdd(&ptr[i], integer_type(0));
    }

    return u.get_value();
  }

  template < typename T >
  static RAJA_DEVICE RAJA_INLINE void set(T* in_ptr, size_t idx, T val)
  {
    using ArrayType = RAJA::detail::AsIntegerArray<T, min_atomic_int_type_size, max_atomic_int_type_size>;
    using integer_type = typename ArrayType::integer_type;

    ArrayType u;
    u.set_value(val);
    auto ptr = reinterpret_cast<integer_type*>(in_ptr + idx);

    for (size_t i = 0; i < u.array_size(); ++i) {
      atomicExch(&ptr[i], u.array[i]);
    }
  }

  static RAJA_DEVICE RAJA_INLINE void fence_acquire()
  {
    __threadfence();
  }

  static RAJA_DEVICE RAJA_INLINE void fence_release()
  {
    __threadfence();
  }
};


// cuda 8 only has shfl primitives for 32 bits while cuda 9 has 32 and 64 bits
constexpr size_t min_shfl_int_type_size = sizeof(unsigned int);
#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
constexpr size_t max_shfl_int_type_size = sizeof(unsigned long long);
#else
constexpr size_t max_shfl_int_type_size = sizeof(unsigned int);
#endif

/*!
 ******************************************************************************
 *
 * \brief Method to shuffle 32b registers in sum reduction for arbitrary type.
 *
 * \Note Returns an undefined value if src lane is inactive (divergence).
 *       Returns this lane's value if src lane is out of bounds or has exited.
 *
 ******************************************************************************
 */
template <typename T>
RAJA_DEVICE RAJA_INLINE T shfl_xor_sync(T var, int laneMask)
{
  RAJA::detail::AsIntegerArray<T, min_shfl_int_type_size, max_shfl_int_type_size> u;
  u.set_value(var);

  for (size_t i = 0; i < u.array_size(); ++i) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
    u.array[i] = ::__shfl_xor_sync(0xffffffffu, u.array[i], laneMask);
#else
    u.array[i] = ::__shfl_xor(u.array[i], laneMask);
#endif
  }
  return u.get_value();
}

template <typename T>
RAJA_DEVICE RAJA_INLINE T shfl_sync(T var, int srcLane)
{
  RAJA::detail::AsIntegerArray<T, min_shfl_int_type_size, max_shfl_int_type_size> u;
  u.set_value(var);

  for (size_t i = 0; i < u.array_size(); ++i) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
    u.array[i] = ::__shfl_sync(0xffffffffu, u.array[i], srcLane);
#else
    u.array[i] = ::__shfl(u.array[i], srcLane);
#endif
  }
  return u.get_value();
}

#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000

template <>
RAJA_DEVICE RAJA_INLINE int shfl_xor_sync<int>(int var, int laneMask)
{
  return ::__shfl_xor_sync(0xffffffffu, var, laneMask);
}

template <>
RAJA_DEVICE RAJA_INLINE unsigned int shfl_xor_sync<unsigned int>(unsigned int var, int laneMask)
{
  return ::__shfl_xor_sync(0xffffffffu, var, laneMask);
}

template <>
RAJA_DEVICE RAJA_INLINE long shfl_xor_sync<long>(long var, int laneMask)
{
  return ::__shfl_xor_sync(0xffffffffu, var, laneMask);
}

template <>
RAJA_DEVICE RAJA_INLINE unsigned long shfl_xor_sync<unsigned long>(unsigned long var, int laneMask)
{
  return ::__shfl_xor_sync(0xffffffffu, var, laneMask);
}

template <>
RAJA_DEVICE RAJA_INLINE long long shfl_xor_sync<long long>(long long var, int laneMask)
{
  return ::__shfl_xor_sync(0xffffffffu, var, laneMask);
}

template <>
RAJA_DEVICE RAJA_INLINE unsigned long long shfl_xor_sync<unsigned long long>(unsigned long long var, int laneMask)
{
  return ::__shfl_xor_sync(0xffffffffu, var, laneMask);
}

template <>
RAJA_DEVICE RAJA_INLINE float shfl_xor_sync<float>(float var, int laneMask)
{
  return ::__shfl_xor_sync(0xffffffffu, var, laneMask);
}

template <>
RAJA_DEVICE RAJA_INLINE double shfl_xor_sync<double>(double var, int laneMask)
{
  return ::__shfl_xor_sync(0xffffffffu, var, laneMask);
}

#else

template <>
RAJA_DEVICE RAJA_INLINE int shfl_xor_sync<int>(int var, int laneMask)
{
  return ::__shfl_xor(var, laneMask);
}

template <>
RAJA_DEVICE RAJA_INLINE float shfl_xor_sync<float>(float var, int laneMask)
{
  return ::__shfl_xor(var, laneMask);
}

#endif


#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000

template <>
RAJA_DEVICE RAJA_INLINE int shfl_sync<int>(int var, int srcLane)
{
  return ::__shfl_sync(0xffffffffu, var, srcLane);
}

template <>
RAJA_DEVICE RAJA_INLINE unsigned int shfl_sync<unsigned int>(unsigned int var, int srcLane)
{
  return ::__shfl_sync(0xffffffffu, var, srcLane);
}

template <>
RAJA_DEVICE RAJA_INLINE long shfl_sync<long>(long var, int srcLane)
{
  return ::__shfl_sync(0xffffffffu, var, srcLane);
}

template <>
RAJA_DEVICE RAJA_INLINE unsigned long shfl_sync<unsigned long>(unsigned long var, int srcLane)
{
  return ::__shfl_sync(0xffffffffu, var, srcLane);
}

template <>
RAJA_DEVICE RAJA_INLINE long long shfl_sync<long long>(long long var, int srcLane)
{
  return ::__shfl_sync(0xffffffffu, var, srcLane);
}

template <>
RAJA_DEVICE RAJA_INLINE unsigned long long shfl_sync<unsigned long long>(unsigned long long var, int srcLane)
{
  return ::__shfl_sync(0xffffffffu, var, srcLane);
}

template <>
RAJA_DEVICE RAJA_INLINE float shfl_sync<float>(float var, int srcLane)
{
  return ::__shfl_sync(0xffffffffu, var, srcLane);
}

template <>
RAJA_DEVICE RAJA_INLINE double shfl_sync<double>(double var, int srcLane)
{
  return ::__shfl_sync(0xffffffffu, var, srcLane);
}

#else

template <>
RAJA_DEVICE RAJA_INLINE int shfl_sync<int>(int var, int srcLane)
{
  return ::__shfl(var, srcLane);
}

template <>
RAJA_DEVICE RAJA_INLINE float shfl_sync<float>(float var, int srcLane)
{
  return ::__shfl(var, srcLane);
}

#endif


//! reduce values in block into thread 0
template <typename Combiner, typename T>
RAJA_DEVICE RAJA_INLINE T warp_reduce(T val, T RAJA_UNUSED_ARG(identity))
{
  int numThreads = blockDim.x * blockDim.y * blockDim.z;

  int threadId = threadIdx.x + blockDim.x * threadIdx.y +
                 (blockDim.x * blockDim.y) * threadIdx.z;

  T temp = val;

  if (numThreads % policy::cuda::WARP_SIZE == 0) {

    // reduce each warp
    for (int i = 1; i < policy::cuda::WARP_SIZE; i *= 2) {
      T rhs = shfl_xor_sync(temp, i);
      Combiner{}(temp, rhs);
    }

  } else {

    // reduce each warp
    for (int i = 1; i < policy::cuda::WARP_SIZE; i *= 2) {
      int srcLane = threadId ^ i;
      T rhs = shfl_sync(temp, srcLane);
      // only add from threads that exist (don't double count own value)
      if (srcLane < numThreads) {
        Combiner{}(temp, rhs);
      }
    }
  }

  return temp;
}

/*!
 * Allreduce values in a warp.
 *
 *
 * This does a butterfly pattern leaving each lane with the full reduction
 *
 */
template <typename Combiner, typename T>
RAJA_DEVICE RAJA_INLINE T warp_allreduce(T val)
{
  T temp = val;

  for (int i = 1; i < policy::cuda::WARP_SIZE; i *= 2) {
    T rhs = __shfl_xor_sync(0xffffffff, temp, i);
    Combiner{}(temp, rhs);
  }

  return temp;
}


//! reduce values in block into thread 0
template <typename Combiner, typename T>
RAJA_DEVICE RAJA_INLINE T block_reduce(T val, T identity)
{
  int numThreads = blockDim.x * blockDim.y * blockDim.z;

  int threadId = threadIdx.x + blockDim.x * threadIdx.y +
                 (blockDim.x * blockDim.y) * threadIdx.z;

  int warpId = threadId % policy::cuda::WARP_SIZE;
  int warpNum = threadId / policy::cuda::WARP_SIZE;

  T temp = val;

  if (numThreads % policy::cuda::WARP_SIZE == 0) {

    // reduce each warp
    for (int i = 1; i < policy::cuda::WARP_SIZE; i *= 2) {
      T rhs = shfl_xor_sync(temp, i);
      Combiner{}(temp, rhs);
    }

  } else {

    // reduce each warp
    for (int i = 1; i < policy::cuda::WARP_SIZE; i *= 2) {
      int srcLane = threadId ^ i;
      T rhs = shfl_sync(temp, srcLane);
      // only add from threads that exist (don't double count own value)
      if (srcLane < numThreads) {
        Combiner{}(temp, rhs);
      }
    }
  }

  // reduce per warp values
  if (numThreads > policy::cuda::WARP_SIZE) {

    static_assert(policy::cuda::MAX_WARPS <= policy::cuda::WARP_SIZE,
        "Max Warps must be less than or equal to Warp Size for this algorithm to work");

    // Need to separate declaration and initialization for clang-cuda
    __shared__ unsigned char tmpsd[sizeof(RAJA::detail::SoAArray<T, policy::cuda::MAX_WARPS>)];

    // Partial placement new: Should call new(tmpsd) here but recasting memory
    // to avoid calling constructor/destructor in shared memory.
    RAJA::detail::SoAArray<T, policy::cuda::MAX_WARPS>* sd =
      reinterpret_cast<RAJA::detail::SoAArray<T, policy::cuda::MAX_WARPS> *>(tmpsd);

    // write per warp values to shared memory
    if (warpId == 0) {
      sd->set(warpNum, temp);
    }

    __syncthreads();

    if (warpNum == 0) {

      // read per warp values
      if (warpId * policy::cuda::WARP_SIZE < numThreads) {
        temp = sd->get(warpId);
      } else {
        temp = identity;
      }

      for (int i = 1; i < policy::cuda::MAX_WARPS; i *= 2) {
        T rhs = shfl_xor_sync(temp, i);
        Combiner{}(temp, rhs);
      }
    }

    __syncthreads();
  }

  return temp;
}

}  // end namespace impl

}  // end namespace cuda

}  // end namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
