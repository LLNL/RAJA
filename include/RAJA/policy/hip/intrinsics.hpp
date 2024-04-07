/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA intrinsics templates for HIP execution.
 *
 *          These methods should work on any platform that supports
 *          HIP devices.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_hip_intrinsics_HPP
#define RAJA_hip_intrinsics_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include <type_traits>

#include <hip/hip_runtime.h>

#include "RAJA/util/macros.hpp"
#include "RAJA/util/SoAArray.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/hip/policy.hpp"


namespace RAJA
{

namespace hip
{

namespace impl
{

/*!
 * \brief Abstracts access to memory using normal memory accesses.
 */
struct AccessorWithFences : RAJA::detail::DefaultAccessor
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
 * \brief Abstracts access to memory using atomic memory accesses.
 *
 * \Note Memory access through this class does not guarantee safe access to a
 *       value that is accessed concurrently by other threads as it may split
 *       memory operations into multiple atomic instructions.
 * \Note Fences used through this class only guarantee ordering, they do not
 *       guarantee visiblity of non-atomic memory operations as it may not
 *       actually flush the cache.
 *
 ******************************************************************************
 */
struct AccessorAvoidingFences
{
  // hip has 32 and 64 bit atomics
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
#if defined(RAJA_USE_HIP_INTRINSICS)
      u.array[i] = __hip_atomic_load(&ptr[i], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
      u.array[i] = atomicAdd(&ptr[i], integer_type(0));
#endif
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
#if defined(RAJA_USE_HIP_INTRINSICS)
      __hip_atomic_store(&ptr[i], u.array[i], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
      atomicExch(&ptr[i], u.array[i]);
#endif
    }
  }

  static RAJA_DEVICE RAJA_INLINE void fence_acquire()
  {
#if defined(RAJA_USE_HIP_INTRINSICS)
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
#else
    __threadfence();
#endif
  }

  static RAJA_DEVICE RAJA_INLINE void fence_release()
  {
#if defined(RAJA_USE_HIP_INTRINSICS)
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    // Wait until all vmem operations complete (s_waitcnt vmcnt(0))
    __builtin_amdgcn_s_waitcnt(/*vmcnt*/ 0 | (/*exp_cnt*/ 0x7 << 4) | (/*lgkmcnt*/ 0xf << 8));
#else
    __threadfence();
#endif
  }
};


// hip only has shfl primitives for 32 bits
constexpr size_t min_shfl_int_type_size = sizeof(unsigned int);
constexpr size_t max_shfl_int_type_size = sizeof(unsigned int);

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
    u.array[i] = ::__shfl_xor(u.array[i], laneMask);
  }
  return u.get_value();
}

template <typename T>
RAJA_DEVICE RAJA_INLINE T shfl_sync(T var, int srcLane)
{
  RAJA::detail::AsIntegerArray<T, min_shfl_int_type_size, max_shfl_int_type_size> u;
  u.set_value(var);

  for (size_t i = 0; i < u.array_size(); ++i) {
    u.array[i] = ::__shfl(u.array[i], srcLane);
  }
  return u.get_value();
}


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


//! reduce values in block into thread 0
template <typename Combiner, typename T>
RAJA_DEVICE RAJA_INLINE T warp_reduce(T val, T RAJA_UNUSED_ARG(identity))
{
  int numThreads = blockDim.x * blockDim.y * blockDim.z;

  int threadId = threadIdx.x + blockDim.x * threadIdx.y +
                 (blockDim.x * blockDim.y) * threadIdx.z;

  T temp = val;

  if (numThreads % policy::hip::WARP_SIZE == 0) {

    // reduce each warp
    for (int i = 1; i < policy::hip::WARP_SIZE; i *= 2) {
      T rhs = shfl_xor_sync(temp, i);
      Combiner{}(temp, rhs);
    }

  } else {

    // reduce each warp
    for (int i = 1; i < policy::hip::WARP_SIZE; i *= 2) {
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

  for (int i = 1; i < policy::hip::WARP_SIZE; i *= 2) {
    T rhs = shfl_xor_sync(temp, i);
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

  int warpId = threadId % policy::hip::WARP_SIZE;
  int warpNum = threadId / policy::hip::WARP_SIZE;

  T temp = val;

  if (numThreads % policy::hip::WARP_SIZE == 0) {

    // reduce each warp
    for (int i = 1; i < policy::hip::WARP_SIZE; i *= 2) {
      T rhs = shfl_xor_sync(temp, i);
      Combiner{}(temp, rhs);
    }

  } else {

    // reduce each warp
    for (int i = 1; i < policy::hip::WARP_SIZE; i *= 2) {
      int srcLane = threadId ^ i;
      T rhs = shfl_sync(temp, srcLane);
      // only add from threads that exist (don't double count own value)
      if (srcLane < numThreads) {
        Combiner{}(temp, rhs);
      }
    }
  }

  // reduce per warp values
  if (numThreads > policy::hip::WARP_SIZE) {

    static_assert(policy::hip::MAX_WARPS <= policy::hip::WARP_SIZE,
        "Max Warps must be less than or equal to Warp Size for this algorithm to work");

    __shared__ unsigned char tmpsd[sizeof(RAJA::detail::SoAArray<T, policy::hip::MAX_WARPS>)];
    RAJA::detail::SoAArray<T, policy::hip::MAX_WARPS>* sd =
      reinterpret_cast<RAJA::detail::SoAArray<T, policy::hip::MAX_WARPS> *>(tmpsd);

    // write per warp values to shared memory
    if (warpId == 0) {
      sd->set(warpNum, temp);
    }

    __syncthreads();

    if (warpNum == 0) {

      // read per warp values
      if (warpId * policy::hip::WARP_SIZE < numThreads) {
        temp = sd->get(warpId);
      } else {
        temp = identity;
      }

      for (int i = 1; i < policy::hip::MAX_WARPS; i *= 2) {
        T rhs = shfl_xor_sync(temp, i);
        Combiner{}(temp, rhs);
      }
    }

    __syncthreads();
  }

  return temp;
}

}  // end namespace impl

}  // end namespace hip

}  // end namespace RAJA

#endif  // closing endif for RAJA_ENABLE_HIP guard

#endif  // closing endif for header file include guard
