/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining atomic class.
 *
 ******************************************************************************
 */

#ifndef RAJA_Atomic_cuda_HXX
#define RAJA_Atomic_cuda_HXX

#if defined(RAJA_ENABLE_CUDA)

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read raja/README-license.txt.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/config.hxx"

#include "RAJA/exec-cuda/MemUtils_CUDA.hxx"

#include <type_traits>

namespace RAJA
{

///
/// Max number of cuda tomic variables.
///
#define RAJA_MAX_ATOMIC_CUDA_VARS 8
#define RAJA_CUDA_ATOMIC_VAR_MAXSIZE sizeof(double)

//
//////////////////////////////////////////////////////////////////////
//
// Atomic policy
//
//////////////////////////////////////////////////////////////////////
//

///
/// Basic cpu use only atomic policy.
///
template < bool Async >
struct cuda_atomic_t {
};

using cuda_atomic = cuda_atomic_t<false>;
using cuda_atomic_async = cuda_atomic_t<true>;

// INTERNAL namespace to encapsulate helper functions
namespace INTERNAL
{
/*!
 ******************************************************************************
 *
 * \brief Method to shuffle 32b registers in sum reduction for arbitrary type.
 *
 ******************************************************************************
 */
template<typename T>
__device__ __forceinline__ T shfl(T var, int laneMask)
{
  const int int_sizeof_T = 
      (sizeof(T) + sizeof(int) - 1) / sizeof(int);
  union {
    T var;
    int arr[int_sizeof_T];
  } Tunion;
  Tunion.var = var;

  for(int i = 0; i < int_sizeof_T; ++i) {
    Tunion.arr[i] = __shfl(Tunion.arr[i], laneMask);
  }
  return Tunion.var;
}

} // end INTERNAL namespace for helper functions

/*!
 ******************************************************************************
 *
 * \brief  Atomic cuda_atomic class specialization.
 *
 * Note: Memory_order defaults to relaxed instead of seq_cst.
 *
 * Note: This class currently uses cuda reduction memory slots, hence adds to
 *       the counts of cuda reductions.
 *
 * Note: Most methods are const because variables capture by value in lambdas
 *       are const.
 *
 ******************************************************************************
 */
template < typename T, bool Async >
class atomic < cuda_atomic_t<Async>, T >
{
public:
  using default_memory_order_t = raja_memory_order_relaxed_t;

  ///
  /// Default constructor default constructs T.
  ///
  __host__
  atomic() noexcept = delete;
  //   : m_myID(getCudaReductionId()),
  //     m_is_copy_host(false)
  // {
  //   getCudaReductionTallyBlock(m_myID,
  //                              (void **)&m_impl_host,
  //                              (void **)&m_impl_device);
  //   m_impl_host[0] = T();
  // }

  ///
  /// Constructor to initialize T with val.
  ///
  __host__
  explicit atomic(T val = T()) noexcept
    : m_myID(getCudaReductionId()),
      m_is_copy_host(false)
  {
    getCudaReductionTallyBlock(m_myID,
                               (void **)&m_impl_host,
                               (void **)&m_impl_device);
    m_impl_host[0] = val;
  }

  ///
  /// Atomic copy constructor.
  ///
  __host__ __device__
  atomic(const atomic& other) noexcept
    : m_impl_device(other.m_impl_device),
      m_impl_host(other.m_impl_host),
      m_myID(other.m_myID),
      m_is_copy_host(true)
  {

  }

  ///
  /// Atomic copy assignment operator deleted.
  ///
  atomic& operator=(const atomic&) = delete;
  atomic& operator=(const atomic&) volatile = delete;

  ///
  /// Atomic destructor frees resources if not a copy.
  ///
  __host__ __device__
  ~atomic() noexcept
  {
#if !defined(__CUDA_ARCH__)
    if (!m_is_copy_host) {
      releaseCudaReductionTallyBlock(m_myID);
      releaseCudaReductionId(m_myID);
    }
#endif
  }

  ///
  /// Atomically assign val, equivalent to store(val).
  ///
  __host__ __device__
  T operator=(T val) const volatile noexcept
  {
#if defined(__CUDA_ARCH__)
    _atomicExch(static_cast<volatile T*>(m_impl_device), val);
#else
    beforeCudaWriteTallyBlock<Async>(m_myID);
    static_cast<volatile T*>(m_impl_host)[0] = val;
#endif
    return val;
  }
  __host__ __device__
  T operator=(T val) const noexcept
  {
#if defined(__CUDA_ARCH__)
    _atomicExch(m_impl_device, val);
#else
    beforeCudaWriteTallyBlock<Async>(m_myID);
    m_impl_host[0] = val;
#endif
    return val;
  }

  ///
  /// Atomic store val.
  ///
  template< typename MEM_ORDER = default_memory_order_t >
  __host__ __device__
  void store(T val, MEM_ORDER m = default_memory_order_t()) const volatile noexcept
  {
#if defined(__CUDA_ARCH__)
    _atomicExch(static_cast<volatile T*>(m_impl_device), val);
#else
    beforeCudaWriteTallyBlock<Async>(m_myID);
    static_cast<volatile T*>(m_impl_host)[0] = val;
#endif
  }
  template< typename MEM_ORDER = default_memory_order_t >
  __host__ __device__
  void store(T val, MEM_ORDER m = default_memory_order_t()) const noexcept
  {
#if defined(__CUDA_ARCH__)
    _atomicExch(m_impl_device, val);
#else
    beforeCudaWriteTallyBlock<Async>(m_myID);
    m_impl_host[0] = val;
#endif
  }

  ///
  /// Atomic load.
  ///
  template< typename MEM_ORDER = default_memory_order_t >
  __host__ __device__
  T load(MEM_ORDER m = default_memory_order_t()) const volatile noexcept
  {
    T current = T();
#if defined(__CUDA_ARCH__)
    current = _atomicCAS(static_cast<volatile T*>(m_impl_device), current, current);
#else
    beforeCudaReadTallyBlock<Async>(m_myID);
    current = static_cast<volatile T*>(m_impl_host)[0];
#endif
    return current;
  }
  template< typename MEM_ORDER = default_memory_order_t >
  __host__ __device__
  T load(MEM_ORDER m = default_memory_order_t()) const noexcept
  {
    T current = T();
#if defined(__CUDA_ARCH__)
    current = _atomicCAS(m_impl_device, current, current);
#else
    beforeCudaReadTallyBlock<Async>(m_myID);
    current = m_impl_host[0];
#endif
    return current;
  }

  ///
  /// Atomically load, equivalent to load().
  ///
  __host__ __device__
  operator T() const volatile noexcept
  {
    T current = T();
#if defined(__CUDA_ARCH__)
    current = _atomicCAS(static_cast<volatile T*>(m_impl_device), current, current);
#else
    beforeCudaReadTallyBlock<Async>(m_myID);
    current = static_cast<volatile T*>(m_impl_host)[0];
#endif
    return current;
  }
  __host__ __device__
  operator T() const noexcept
  {
    T current = T();
#if defined(__CUDA_ARCH__)
    current = _atomicCAS(m_impl_device, current, current);
#else
    beforeCudaReadTallyBlock<Async>(m_myID);
    current = m_impl_host[0];
#endif
    return current;
  }

  ///
  /// Atomically loads what is stored while replacing it with val.
  /// Returns what was previously stored.
  ///
  template< typename MEM_ORDER = default_memory_order_t >
  __device__
  T exchange(T val, MEM_ORDER m = default_memory_order_t()) const volatile noexcept
  {
    return _atomicExch(static_cast<volatile T*>(m_impl_device), val);
  }
  template< typename MEM_ORDER = default_memory_order_t >
  __device__
  T exchange(T val, MEM_ORDER m = default_memory_order_t()) const noexcept
  {
    return _atomicExch(m_impl_device, val);
  }

  ///
  /// Atomically compares what is stored with expected and if same replaces what is stored with val other wise loads what is stored into expected.
  /// Returns true if exchange succeeded, false otherwise, expected is overwritten with what was stored.
  /// Note that weak may fail even if the stored value == expected, but may perform better in a loop on some platforms.
  ///
  template< typename MEM_ORDER = default_memory_order_t >
  __device__
  bool compare_exchange_weak(T& expected, T val, MEM_ORDER m = default_memory_order_t()) const volatile noexcept
  {
    T old_exp = expected;
    expected = _atomicCAS(static_cast<volatile T*>(m_impl_device), expected, val);
    return old_exp == expected;
  }
  template< typename MEM_ORDER = default_memory_order_t >
  __device__
  bool compare_exchange_weak(T& expected, T val, MEM_ORDER m = default_memory_order_t()) const noexcept
  {
    T old_exp = expected;
    expected = _atomicCAS(m_impl_device, expected, val);
    return old_exp == expected;
  }
  template< typename MEM_ORDER = default_memory_order_t >
  __device__
  bool compare_exchange_strong(T& expected, T val, MEM_ORDER m = default_memory_order_t()) const volatile noexcept
  {
    T old_exp = expected;
    expected = _atomicCAS(static_cast<volatile T*>(m_impl_device), expected, val);
    return old_exp == expected;
  }
  template< typename MEM_ORDER = default_memory_order_t >
  __device__
  bool compare_exchange_strong(T& expected, T val, MEM_ORDER m = default_memory_order_t()) const noexcept
  {
    T old_exp = expected;
    expected = _atomicCAS(m_impl_device, expected, val);
    return old_exp == expected;
  }
  template< typename MEM_ORDER_0, typename MEM_ORDER_1 >
  __device__
  bool compare_exchange_weak(T& expected, T val, MEM_ORDER_0 m0, MEM_ORDER_1 m1) const volatile noexcept
  {
    T old_exp = expected;
    expected = _atomicCAS(static_cast<volatile T*>(m_impl_device), expected, val);
    return old_exp == expected;
  }
  template< typename MEM_ORDER_0, typename MEM_ORDER_1 >
  __device__
  bool compare_exchange_weak(T& expected, T val, MEM_ORDER_0 m0, MEM_ORDER_1 m1) const noexcept
  {
    T old_exp = expected;
    expected = _atomicCAS(m_impl_device, expected, val);
    return old_exp == expected;
  }
  template< typename MEM_ORDER_0, typename MEM_ORDER_1 >
  __device__
  bool compare_exchange_strong(T& expected, T val, MEM_ORDER_0 m0, MEM_ORDER_1 m1) const volatile noexcept
  {
    T old_exp = expected;
    expected = _atomicCAS(static_cast<volatile T*>(m_impl_device), expected, val);
    return old_exp == expected;
  }
  template< typename MEM_ORDER_0, typename MEM_ORDER_1 >
  __device__
  bool compare_exchange_strong(T& expected, T val, MEM_ORDER_0 m0, MEM_ORDER_1 m1) const noexcept
  {
    T old_exp = expected;
    expected = _atomicCAS(m_impl_device, expected, val);
    return old_exp == expected;
  }

  ///
  /// Atomically operate on the stored value and return the value as it was before this operation.
  ///
  template< typename MEM_ORDER = default_memory_order_t >
  __device__
  T fetch_add(T val, MEM_ORDER m = default_memory_order_t()) const volatile noexcept
  {
    return _atomicAdd(static_cast<volatile T*>(m_impl_device), val);
  }
  template< typename MEM_ORDER = default_memory_order_t >
  __device__
  T fetch_add(T val, MEM_ORDER m = default_memory_order_t()) const noexcept
  {
    return _atomicAdd(m_impl_device, val);
  }
  template< typename MEM_ORDER = default_memory_order_t >
  __device__
  T fetch_sub(T val, MEM_ORDER m = default_memory_order_t()) const volatile noexcept
  {
    return _atomicSub(static_cast<volatile T*>(m_impl_device), val);
  }
  template< typename MEM_ORDER = default_memory_order_t >
  __device__
  T fetch_sub(T val, MEM_ORDER m = default_memory_order_t()) const noexcept
  {
    return _atomicSub(m_impl_device, val);
  }
  template< typename MEM_ORDER = default_memory_order_t >
  __device__
  T fetch_and(T val, MEM_ORDER m = default_memory_order_t()) const volatile noexcept
  {
    return _atomicAnd(static_cast<volatile T*>(m_impl_device), val);
  }
  template< typename MEM_ORDER = default_memory_order_t >
  __device__
  T fetch_and(T val, MEM_ORDER m = default_memory_order_t()) const noexcept
  {
    return _atomicAnd(m_impl_device, val);
  }
  template< typename MEM_ORDER = default_memory_order_t >
  __device__
  T fetch_or(T val, MEM_ORDER m = default_memory_order_t()) const volatile noexcept
  {
    return _atomicOr(static_cast<volatile T*>(m_impl_device), val);
  }
  template< typename MEM_ORDER = default_memory_order_t >
  __device__
  T fetch_or(T val, MEM_ORDER m = default_memory_order_t()) const noexcept
  {
    return _atomicOr(m_impl_device, val);
  }
  template< typename MEM_ORDER = default_memory_order_t >
  __device__
  T fetch_xor(T val, MEM_ORDER m = default_memory_order_t()) const volatile noexcept
  {
    return _atomicXor(static_cast<volatile T*>(m_impl_device), val);
  }
  template< typename MEM_ORDER = default_memory_order_t >
  __device__
  T fetch_xor(T val, MEM_ORDER m = default_memory_order_t()) const noexcept
  {
    return _atomicXor(m_impl_device, val);
  }

  ///
  /// Atomic min operator, returns the previously stored value
  ///
  template< typename MEM_ORDER = default_memory_order_t >
  __device__
  T fetch_min(T val, MEM_ORDER m = default_memory_order_t()) const volatile noexcept
  {
    return _atomicMin(static_cast<volatile T*>(m_impl_device), val);
  }
  template< typename MEM_ORDER = default_memory_order_t >
  __device__
  T fetch_min(T val, MEM_ORDER m = default_memory_order_t()) const noexcept
  {
    return _atomicMin(m_impl_device, val);
  }

  ///
  /// Atomic max operator, returns the previously stored value
  ///
  template< typename MEM_ORDER = default_memory_order_t >
  __device__
  T fetch_max(T val, MEM_ORDER m = default_memory_order_t()) const volatile noexcept
  {
    return _atomicMax(static_cast<volatile T*>(m_impl_device), val);
  }
  template< typename MEM_ORDER = default_memory_order_t >
  __device__
  T fetch_max(T val, MEM_ORDER m = default_memory_order_t()) const noexcept
  {
    return _atomicMax(m_impl_device, val);
  }

  ///
  /// Atomic pre-fix operators. Equivalent to fetch_op(1) op 1
  /// Warp aggregated.
  ///
  __device__
  T operator++() const volatile noexcept
  {
    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;
    int laneId = threadId % WARP_SIZE;
    int mask = __ballot(1);
    int first = __ffs(mask) - 1;
    int ninc = __popc(mask);
    int linc = __popc(mask & ((1 << laneId) - 1)) + 1;
    T val;
    if (laneId == first) {
      val = _atomicAdd(static_cast<volatile T*>(m_impl_device), static_cast<T>(ninc));
    }
    return INTERNAL::shfl(val, first) + static_cast<T>(linc);
  }
  __device__
  T operator++() const noexcept
  {
    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;
    int laneId = threadId % WARP_SIZE;
    int mask = __ballot(1);
    int first = __ffs(mask) - 1;
    int ninc = __popc(mask);
    int linc = __popc(mask & ((1 << laneId) - 1)) + 1;
    T val;
    if (laneId == first) {
      val = _atomicAdd(m_impl_device, static_cast<T>(ninc));
    }
    return INTERNAL::shfl(val, first) + static_cast<T>(linc);
  }
  __device__
  T operator--() const volatile noexcept
  {
    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;
    int laneId = threadId % WARP_SIZE;
    int mask = __ballot(1);
    int first = __ffs(mask) - 1;
    int ninc = __popc(mask);
    int linc = __popc(mask & ((1 << laneId) - 1)) + 1;
    T val;
    if (laneId == first) {
      val = _atomicSub(static_cast<volatile T*>(m_impl_device), static_cast<T>(ninc));
    }
    return INTERNAL::shfl(val, first) - static_cast<T>(linc);
  }
  __device__
  T operator--() const noexcept
  {
    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;
    int laneId = threadId % WARP_SIZE;
    int mask = __ballot(1);
    int first = __ffs(mask) - 1;
    int ninc = __popc(mask);
    int linc = __popc(mask & ((1 << laneId) - 1)) + 1;
    T val;
    if (laneId == first) {
      val = _atomicSub(m_impl_device, static_cast<T>(ninc));
    }
    return INTERNAL::shfl(val, first) - static_cast<T>(linc);
  }

  ///
  /// Atomic post-fix operators. Equivalent to fetch_op(1)
  /// Warp aggregated.
  ///
  __device__
  T operator++(int) const volatile noexcept
  {
    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;
    int laneId = threadId % WARP_SIZE;
    int mask = __ballot(1);
    int first = __ffs(mask) - 1;
    int ninc = __popc(mask);
    int linc = __popc(mask & ((1 << laneId) - 1));
    T val;
    if (laneId == first) {
      val = _atomicAdd(static_cast<volatile T*>(m_impl_device), static_cast<T>(ninc));
    }
    return INTERNAL::shfl(val, first) + static_cast<T>(linc);
  }
  __device__
  T operator++(int) const noexcept
  {
    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;
    int laneId = threadId % WARP_SIZE;
    int mask = __ballot(1);
    int first = __ffs(mask) - 1;
    int ninc = __popc(mask);
    int linc = __popc(mask & ((1 << laneId) - 1));
    T val;
    if (laneId == first) {
      val = _atomicAdd(m_impl_device, static_cast<T>(ninc));
    }
    return INTERNAL::shfl(val, first) + static_cast<T>(linc);
  }
  __device__
  T operator--(int) const volatile noexcept
  {
    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;
    int laneId = threadId % WARP_SIZE;
    int mask = __ballot(1);
    int first = __ffs(mask) - 1;
    int ninc = __popc(mask);
    int linc = __popc(mask & ((1 << laneId) - 1));
    T val;
    if (laneId == first) {
      val = _atomicSub(static_cast<volatile T*>(m_impl_device), static_cast<T>(ninc));
    }
    return INTERNAL::shfl(val, first) - static_cast<T>(linc);
  }
  __device__
  T operator--(int) const noexcept
  {
    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;
    int laneId = threadId % WARP_SIZE;
    int mask = __ballot(1);
    int first = __ffs(mask) - 1;
    int ninc = __popc(mask);
    int linc = __popc(mask & ((1 << laneId) - 1));
    T val;
    if (laneId == first) {
      val = _atomicSub(m_impl_device, static_cast<T>(ninc));
    }
    return INTERNAL::shfl(val, first) - static_cast<T>(linc);
  }

  ///
  /// Atomic operators. Equivalent to fetch_op(val) op val
  ///
  __device__
  T operator+=(T val) const volatile noexcept
  {
    return _atomicAdd(static_cast<volatile T*>(m_impl_device), val) + val;
  }
  __device__
  T operator+=(T val) const noexcept
  {
    return _atomicAdd(m_impl_device, val) + val;
  }
  __device__
  T operator-=(T val) const volatile noexcept
  {
    return _atomicSub(static_cast<volatile T*>(m_impl_device), val) - val;
  }
  __device__
  T operator-=(T val) const noexcept
  {
    return _atomicSub(m_impl_device, val) - val;
  }
  __device__
  T operator&=(T val) const volatile noexcept
  {
    return _atomicAnd(static_cast<volatile T*>(m_impl_device), val) & val;
  }
  __device__
  T operator&=(T val) const noexcept
  {
    return _atomicAnd(m_impl_device, val) & val;
  }
  __device__
  T operator|=(T val) const volatile noexcept
  {
    return _atomicOr(static_cast<volatile T*>(m_impl_device), val) | val;
  }
  __device__
  T operator|=(T val) const noexcept
  {
    return _atomicOr(m_impl_device, val) | val;
  }
  __device__
  T operator^=(T val) const volatile noexcept
  {
    return _atomicXor(static_cast<volatile T*>(m_impl_device), val) ^ val;
  }
  __device__
  T operator^=(T val) const noexcept
  {
    return _atomicXor(m_impl_device, val) ^ val;
  }

private:
  ///
  /// Implementation via T pointer on host and device.
  ///
  T* m_impl_device = nullptr;
  T* m_impl_host = nullptr;

  ///
  /// My cuda reduction variable ID.
  ///
  int m_myID = -1;

  ///
  /// Remember if are copy to free m_impl_device.
  ///
  const bool m_is_copy_host;

  static_assert(std::is_arithmetic<T>::value,
                "Error: type must be arithmetic");
  static_assert(sizeof(T) <= RAJA_CUDA_ATOMIC_VAR_MAXSIZE,
                "Error: type must be of size <= "
                RAJA_STRINGIFY_MACRO(RAJA_CUDA_ATOMIC_VAR_MAXSIZE));
};

}  // closing brace for RAJA namespace

#endif  // Closing endif for defined(RAJA_ENABLE_CUDA)

#endif  // closing endif for header file include guard
