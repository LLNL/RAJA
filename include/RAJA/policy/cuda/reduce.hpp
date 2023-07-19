/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA reduction templates for CUDA execution.
 *
 *          These methods should work on any platform that supports
 *          CUDA devices.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_cuda_reduce_HPP
#define RAJA_cuda_reduce_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include <cuda.h>

#include "RAJA/util/macros.hpp"
#include "RAJA/util/SoAArray.hpp"
#include "RAJA/util/SoAPtr.hpp"
#include "RAJA/util/basic_mempool.hpp"
#include "RAJA/util/mutex.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/detail/reduce.hpp"
#include "RAJA/pattern/reduce.hpp"

#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"

#if defined(RAJA_ENABLE_DESUL_ATOMICS)
  #include "RAJA/policy/desul/atomic.hpp"
#else
  #include "RAJA/policy/cuda/atomic.hpp"
#endif

#include "RAJA/policy/cuda/policy.hpp"
#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"

namespace RAJA
{

namespace reduce
{

namespace cuda
{
//! atomic operator version of Combiner object
template <typename Combiner>
struct atomic;

template <typename T>
struct atomic<sum<T>> {
  RAJA_DEVICE RAJA_INLINE void operator()(T& val, const T v)
  {
    RAJA::atomicAdd<T>(RAJA::cuda_atomic{}, &val, v);
  }
};

template <typename T>
struct atomic<min<T>> {
  RAJA_DEVICE RAJA_INLINE void operator()(T& val, const T v)
  {
    RAJA::atomicMin<T>(RAJA::cuda_atomic{}, &val, v);
  }
};

template <typename T>
struct atomic<max<T>> {
  RAJA_DEVICE RAJA_INLINE void operator()(T& val, const T v)
  {
    RAJA::atomicMax<T>(RAJA::cuda_atomic{}, &val, v);
  }
};

template <typename T>
struct cuda_atomic_available {
  static constexpr const bool value =
      (std::is_integral<T>::value && (4 == sizeof(T) || 8 == sizeof(T))) ||
      std::is_same<T, float>::value || std::is_same<T, double>::value;
};

}  // namespace cuda

}  // namespace reduce

namespace cuda
{

namespace impl
{

/*!
 * \brief Abstracts T into an equal or greater size array of integers whose
 * size is between min_integer_type_size and max_interger_type_size inclusive.
 */
template <typename T,
          size_t min_integer_type_size = 1,
          size_t max_integer_type_size = sizeof(long long)>
union AsIntegerArray {

  static_assert(min_integer_type_size <= max_integer_type_size,
                "incompatible min and max integer type size");
  using integer_type = typename std::conditional<
      ((alignof(T) >= alignof(long long) &&
        sizeof(long long) <= max_integer_type_size) ||
       sizeof(long) < min_integer_type_size),
      long long,
      typename std::conditional<
          ((alignof(T) >= alignof(long) &&
            sizeof(long) <= max_integer_type_size) ||
           sizeof(int) < min_integer_type_size),
          long,
          typename std::conditional<
              ((alignof(T) >= alignof(int) &&
                sizeof(int) <= max_integer_type_size) ||
               sizeof(short) < min_integer_type_size),
              int,
              typename std::conditional<
                  ((alignof(T) >= alignof(short) &&
                    sizeof(short) <= max_integer_type_size) ||
                   sizeof(char) < min_integer_type_size),
                  short,
                  typename std::conditional<
                      ((alignof(T) >= alignof(char) &&
                        sizeof(char) <= max_integer_type_size)),
                      char,
                      void>::type>::type>::type>::type>::type;
  static_assert(!std::is_same<integer_type, void>::value,
                "could not find a compatible integer type");
  static_assert(sizeof(integer_type) >= min_integer_type_size,
                "integer_type smaller than min integer type size");
  static_assert(sizeof(integer_type) <= max_integer_type_size,
                "integer_type greater than max integer type size");

  static constexpr size_t num_integer_type =
      (sizeof(T) + sizeof(integer_type) - 1) / sizeof(integer_type);

  T value;
  integer_type array[num_integer_type];

  RAJA_HOST_DEVICE constexpr AsIntegerArray(T value_) : value(value_){};

  RAJA_HOST_DEVICE constexpr size_t array_size() const
  {
    return num_integer_type;
  }
};

// cuda 8 only has shfl primitives for 32 bits while cuda 9 has 32 and 64 bits
constexpr const size_t min_shfl_int_type_size = sizeof(int);
#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
constexpr const size_t max_shfl_int_type_size = sizeof(long long);
#else
constexpr const size_t max_shfl_int_type_size = sizeof(int);
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
  AsIntegerArray<T, min_shfl_int_type_size, max_shfl_int_type_size> u(var);

  for (size_t i = 0; i < u.array_size(); ++i) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
    u.array[i] = ::__shfl_xor_sync(0xffffffffu, u.array[i], laneMask);
#else
    u.array[i] = ::__shfl_xor(u.array[i], laneMask);
#endif
  }
  return u.value;
}

template <typename T>
RAJA_DEVICE RAJA_INLINE T shfl_sync(T var, int srcLane)
{
  AsIntegerArray<T, min_shfl_int_type_size, max_shfl_int_type_size> u(var);

  for (size_t i = 0; i < u.array_size(); ++i) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
    u.array[i] = ::__shfl_sync(0xffffffffu, u.array[i], srcLane);
#else
    u.array[i] = ::__shfl(u.array[i], srcLane);
#endif
  }
  return u.value;
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
    RAJA::detail::SoAArray<T, policy::cuda::MAX_WARPS> * sd = reinterpret_cast<RAJA::detail::SoAArray<T, policy::cuda::MAX_WARPS> *>(tmpsd);

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

//! reduce values in grid into thread 0 of last running block
//  returns true if put reduced value in val
template <typename Combiner, typename T, typename TempIterator>
RAJA_DEVICE RAJA_INLINE bool grid_reduce(T& val,
                                         T identity,
                                         TempIterator device_mem,
                                         unsigned int* device_count)
{
  int numBlocks = gridDim.x * gridDim.y * gridDim.z;
  int numThreads = blockDim.x * blockDim.y * blockDim.z;
  unsigned int wrap_around = numBlocks - 1;

  int blockId = blockIdx.x + gridDim.x * blockIdx.y +
                (gridDim.x * gridDim.y) * blockIdx.z;

  int threadId = threadIdx.x + blockDim.x * threadIdx.y +
                 (blockDim.x * blockDim.y) * threadIdx.z;

  T temp = block_reduce<Combiner>(val, identity);

  // one thread per block writes to device_mem
  bool lastBlock = false;
  if (threadId == 0) {
    device_mem.set(blockId, temp);
    // ensure write visible to all threadblocks
    __threadfence();
    // increment counter, (wraps back to zero if old count == wrap_around)
    unsigned int old_count = ::atomicInc(device_count, wrap_around);
    lastBlock = (old_count == wrap_around);
  }

  // returns non-zero value if any thread passes in a non-zero value
  lastBlock = __syncthreads_or(lastBlock);

  // last block accumulates values from device_mem
  if (lastBlock) {
    temp = identity;

    for (int i = threadId; i < numBlocks; i += numThreads) {
      Combiner{}(temp, device_mem.get(i));
    }

    temp = block_reduce<Combiner>(temp, identity);

    // one thread returns value
    if (threadId == 0) {
      val = temp;
    }
  }

  return lastBlock && threadId == 0;
}

namespace expt {

template <typename Combiner, typename T>
RAJA_DEVICE RAJA_INLINE T block_reduce(T val, T identity)
{
  int numThreads = blockDim.x * blockDim.y * blockDim.z;

  int threadId = threadIdx.x + blockDim.x * threadIdx.y +
                 (blockDim.x * blockDim.y) * threadIdx.z;

  int warpId = threadId % RAJA::policy::cuda::WARP_SIZE;
  int warpNum = threadId / RAJA::policy::cuda::WARP_SIZE;

  T temp = val;

  if (numThreads % RAJA::policy::cuda::WARP_SIZE == 0) {

    // reduce each warp
    for (int i = 1; i < RAJA::policy::cuda::WARP_SIZE; i *= 2) {
      T rhs = RAJA::cuda::impl::shfl_xor_sync(temp, i);
      temp = Combiner{}(temp, rhs);
    }

  } else {

    // reduce each warp
    for (int i = 1; i < RAJA::policy::cuda::WARP_SIZE; i *= 2) {
      int srcLane = threadId ^ i;
      T rhs = RAJA::cuda::impl::shfl_sync(temp, srcLane);
      // only add from threads that exist (don't double count own value)
      if (srcLane < numThreads) {
        temp = Combiner{}(temp, rhs);
      }
    }
  }

  static_assert(RAJA::policy::cuda::MAX_WARPS <= RAJA::policy::cuda::WARP_SIZE,
               "Max Warps must be less than or equal to Warp Size for this algorithm to work");

  // reduce per warp values
  if (numThreads > RAJA::policy::cuda::WARP_SIZE) {

    // Need to separate declaration and initialization for clang-cuda
    __shared__ unsigned char tmpsd[sizeof(RAJA::detail::SoAArray<T, RAJA::policy::cuda::MAX_WARPS>)];

    // Partial placement new: Should call new(tmpsd) here but recasting memory
    // to avoid calling constructor/destructor in shared memory.
    RAJA::detail::SoAArray<T, RAJA::policy::cuda::MAX_WARPS> * sd = reinterpret_cast<RAJA::detail::SoAArray<T, RAJA::policy::cuda::MAX_WARPS> *>(tmpsd);

    // write per warp values to shared memory
    if (warpId == 0) {
      sd->set(warpNum, temp);
    }

    __syncthreads();

    if (warpNum == 0) {

      // read per warp values
      if (warpId * RAJA::policy::cuda::WARP_SIZE < numThreads) {
        temp = sd->get(warpId);
      } else {
        temp = identity;
      }

      for (int i = 1; i < RAJA::policy::cuda::MAX_WARPS; i *= 2) {
        T rhs = RAJA::cuda::impl::shfl_xor_sync(temp, i);
        temp = Combiner{}(temp, rhs);
      }
    }

    __syncthreads();
  }

  return temp;
}


template <typename OP, typename T>
RAJA_DEVICE RAJA_INLINE bool grid_reduce(RAJA::expt::detail::Reducer<OP, T>& red) {

  int numBlocks = gridDim.x * gridDim.y * gridDim.z;
  int numThreads = blockDim.x * blockDim.y * blockDim.z;
  unsigned int wrap_around = numBlocks - 1;

  int blockId = blockIdx.x + gridDim.x * blockIdx.y +
                (gridDim.x * gridDim.y) * blockIdx.z;

  int threadId = threadIdx.x + blockDim.x * threadIdx.y +
                 (blockDim.x * blockDim.y) * threadIdx.z;

  T temp = block_reduce<OP>(red.val, OP::identity());

  // one thread per block writes to device_mem
  bool lastBlock = false;
  if (threadId == 0) {
    red.device_mem.set(blockId, temp);
    // ensure write visible to all threadblocks
    __threadfence();
    // increment counter, (wraps back to zero if old count == wrap_around)
    unsigned int old_count = ::atomicInc(red.device_count, wrap_around);
    lastBlock = (old_count == wrap_around);
  }

  // returns non-zero value if any thread passes in a non-zero value
  lastBlock = __syncthreads_or(lastBlock);

  // last block accumulates values from device_mem
  if (lastBlock) {
    temp = OP::identity();

    for (int i = threadId; i < numBlocks; i += numThreads) {
      temp = OP{}(temp, red.device_mem.get(i));
    }

    temp = block_reduce<OP>(temp, OP::identity());

    // one thread returns value
    if (threadId == 0) {
      *(red.devicetarget) = temp;
    }
  }

  return lastBlock && threadId == 0;
}

} //  namespace expt

//! reduce values in grid into thread 0 of last running block
//  returns true if put reduced value in val
template <typename Combiner, typename T>
RAJA_DEVICE RAJA_INLINE bool grid_reduce_atomic(T& val,
                                                T identity,
                                                T* device_mem,
                                                unsigned int* device_count)
{
  int numBlocks = gridDim.x * gridDim.y * gridDim.z;
  unsigned int wrap_around = numBlocks + 1;

  int threadId = threadIdx.x + blockDim.x * threadIdx.y +
                 (blockDim.x * blockDim.y) * threadIdx.z;

  // one thread in first block initializes device_mem
  if (threadId == 0) {
    unsigned int old_val = ::atomicCAS(device_count, 0u, 1u);
    if (old_val == 0u) {
      device_mem[0] = identity;
      __threadfence();
      ::atomicAdd(device_count, 1u);
    }
  }

  T temp = block_reduce<Combiner>(val, identity);

  // one thread per block performs atomic on device_mem
  bool lastBlock = false;
  if (threadId == 0) {
    // thread waits for device_mem to be initialized
    while (static_cast<volatile unsigned int*>(device_count)[0] < 2u)
      ;
    __threadfence();
    RAJA::reduce::cuda::atomic<Combiner>{}(device_mem[0], temp);
    __threadfence();
    // increment counter, (wraps back to zero if old count == wrap_around)
    unsigned int old_count = ::atomicInc(device_count, wrap_around);
    lastBlock = (old_count == wrap_around);

    // last block gets value from device_mem
    if (lastBlock) {
      val = device_mem[0];
    }
  }

  return lastBlock;
}

}  // namespace impl

//! Object that manages pinned memory buffers for reduction results
//  use one per reducer object
template <typename T>
class PinnedTally
{
public:
  //! Object put in Pinned memory with value and pointer to next Node
  struct Node {
    Node* next;
    ::RAJA::resources::Resource res;
    T value;
  };
  //! Object per resource to keep track of pinned memory nodes
  struct ResourceNode {
    ResourceNode* next;
    ::RAJA::resources::Cuda cuda_res;
    Node* node_list;
  };

  //! Iterator over resources used by reducer
  class ResourceIterator
  {
  public:
    ResourceIterator() = delete;

    ResourceIterator(ResourceNode* rn) : m_rn(rn) {}

    const ResourceIterator& operator++()
    {
      m_rn = m_rn->next;
      return *this;
    }

    ResourceIterator operator++(int)
    {
      ResourceIterator ret = *this;
      this->operator++();
      return ret;
    }

    ::RAJA::resources::Cuda& operator*() { return m_rn->cuda_res; }

    bool operator==(const ResourceIterator& rhs) const
    {
      return m_rn == rhs.m_rn;
    }

    bool operator!=(const ResourceIterator& rhs) const
    {
      return !this->operator==(rhs);
    }

  private:
    ResourceNode* m_rn;
  };

  //! Iterator over all values generated by reducer
  class ResourceNodeIterator
  {
  public:
    ResourceNodeIterator() = delete;

    ResourceNodeIterator(ResourceNode* rn, Node* n) : m_rn(rn), m_n(n) {}

    const ResourceNodeIterator& operator++()
    {
      if (m_n->next) {
        m_n = m_n->next;
      } else if (m_rn->next) {
        m_rn = m_rn->next;
        m_n = m_rn->node_list;
      } else {
        m_rn = nullptr;
        m_n = nullptr;
      }
      return *this;
    }

    ResourceNodeIterator operator++(int)
    {
      ResourceNodeIterator ret = *this;
      this->operator++();
      return ret;
    }

    T& operator*() { return m_n->value; }

    bool operator==(const ResourceNodeIterator& rhs) const
    {
      return m_n == rhs.m_n;
    }

    bool operator!=(const ResourceNodeIterator& rhs) const
    {
      return !this->operator==(rhs);
    }

  private:
    ResourceNode* m_rn;
    Node* m_n;
  };

  PinnedTally() : resource_list(nullptr) {}

  PinnedTally(const PinnedTally&) = delete;

  //! get begin iterator over resources
  ResourceIterator resourceBegin() { return {resource_list}; }

  //! get end iterator over resources
  ResourceIterator resourceEnd() { return {nullptr}; }

  //! get begin iterator over values
  ResourceNodeIterator begin()
  {
    return {resource_list, resource_list ? resource_list->node_list : nullptr};
  }

  //! get end iterator over values
  ResourceNodeIterator end() { return {nullptr, nullptr}; }

  //! get new value for use in resource
  Node* new_value(::RAJA::resources::Cuda& cuda_res, ::RAJA::resources::Resource& res)
  {
#if defined(RAJA_ENABLE_OPENMP)
    lock_guard<omp::mutex> lock(m_mutex);
#endif
    ResourceNode* rn = resource_list;
    while (rn) {
      if (rn->cuda_res.get_stream() == cuda_res.get_stream()) break;
      rn = rn->next;
    }
    if (!rn) {
      rn = new ResourceNode{resource_list, cuda_res, nullptr};
      resource_list = rn;
    }
    Node* n_mem = res.template allocate<Node>(1, ::RAJA::resources::MemoryAccess::Pinned);
    rn->node_list = new(n_mem) Node{rn->node_list, res, T{}};
    return rn->node_list;
  }

  //! synchronize all resources used
  void synchronize_resources()
  {
    auto end = resourceEnd();
    for (auto r = resourceBegin(); r != end; ++r) {
      ::RAJA::cuda::synchronize(*r);
    }
  }

  //! all values used in all resources
  void free_list()
  {
    while (resource_list) {
      ResourceNode* rn = resource_list;
      while (rn->node_list) {
        Node* n = rn->node_list;
        rn->node_list = n->next;
        auto res{std::move(n->res)};
        n->~Node();
        res.deallocate(n, ::RAJA::resources::MemoryAccess::Pinned);
      }
      resource_list = rn->next;
      delete rn;
    }
  }

  ~PinnedTally() { free_list(); }

#if defined(RAJA_ENABLE_OPENMP)
  omp::mutex m_mutex;
#endif

private:
  ResourceNode* resource_list;
};

//
//////////////////////////////////////////////////////////////////////
//
// Reduction classes.
//
//////////////////////////////////////////////////////////////////////
//

//! Reduction data for Cuda Offload -- stores value, host pointer, and device
//! pointer
template <typename Combiner, typename T>
struct Reduce_Data {

  mutable T value;
  T identity;
  unsigned int* device_count;
  RAJA::detail::SoAPtr<T> device;
  bool own_device_ptr;

  Reduce_Data() : Reduce_Data(T(), T()){};

  /*! \brief create from a default value and offload information
   *
   *  allocates PinnedTally to hold device values
   */

  Reduce_Data(T initValue, T identity_)
      : value{initValue},
        identity{identity_},
        device_count{nullptr},
        device{},
        own_device_ptr{false}
  {
  }

  void reset(T initValue, T identity_ = T())
  {
    value = initValue;
    identity = identity_;
    device_count = nullptr;
    own_device_ptr = false;
  }

  RAJA_HOST_DEVICE
  Reduce_Data(const Reduce_Data& other)
      : value{other.identity},
        identity{other.identity},
        device_count{other.device_count},
        device{other.device},
        own_device_ptr{false}
  {
  }

  Reduce_Data& operator=(const Reduce_Data&) = default;

  //! initialize output to identity to ensure never read
  //  uninitialized memory
  void init_grid_val(T* output) { *output = identity; }

  //! reduce values in grid to single value, store in output
  RAJA_DEVICE
  void grid_reduce(T* output)
  {
    T temp = value;

    if (impl::grid_reduce<Combiner>(temp, identity, device, device_count)) {
      *output = temp;
    }
  }

  //! check and setup for device
  //  allocate device pointers and get a new result buffer from the pinned tally
  bool setupForDevice(::RAJA::resources::Resource& res)
  {
    bool act = !device.allocated() && setupReducers();
    if (act) {
      cuda_dim_t gridDim = currentGridDim();
      size_t numBlocks = gridDim.x * gridDim.y * gridDim.z;
      device.allocate(numBlocks, res);
      device_count = static_cast<unsigned int*>(
          res.calloc(sizeof(unsigned int), ::RAJA::resources::MemoryAccess::Device));
      own_device_ptr = true;
    }
    return act;
  }

  //! if own resources teardown device setup
  //  free device pointers
  bool teardownForDevice(::RAJA::resources::Resource& res)
  {
    bool act = own_device_ptr;
    if (act) {
      device.deallocate(res);
      res.deallocate(device_count, ::RAJA::resources::MemoryAccess::Device);
      device_count = nullptr;
      own_device_ptr = false;
    }
    return act;
  }
};


//! Reduction data for Cuda Offload -- stores value, host pointer
template <typename Combiner, typename T>
struct ReduceAtomic_Data {

  mutable T value;
  T identity;
  unsigned int* device_count;
  T* device;
  bool own_device_ptr;

  ReduceAtomic_Data() : ReduceAtomic_Data(T(), T()){};

  ReduceAtomic_Data(T initValue, T identity_)
      : value{initValue},
        identity{identity_},
        device_count{nullptr},
        device{nullptr},
        own_device_ptr{false}
  {
  }

  void reset(T initValue, T identity_ = Combiner::identity())
  {
    value = initValue;
    identity = identity_;
    device_count = nullptr;
    device = nullptr;
    own_device_ptr = false;
  }

  RAJA_HOST_DEVICE
  ReduceAtomic_Data(const ReduceAtomic_Data& other)
      : value{other.identity},
        identity{other.identity},
        device_count{other.device_count},
        device{other.device},
        own_device_ptr{false}
  {
  }

  ReduceAtomic_Data& operator=(const ReduceAtomic_Data&) = default;

  //! initialize output to identity to ensure never read
  //  uninitialized memory
  void init_grid_val(T* output) { *output = identity; }

  //! reduce values in grid to single value, store in output
  RAJA_DEVICE
  void grid_reduce(T* output)
  {
    T temp = value;

    if (impl::grid_reduce_atomic<Combiner>(
            temp, identity, device, device_count)) {
      *output = temp;
    }
  }

  //! check and setup for device
  //  allocate device pointers and get a new result buffer from the pinned tally
  bool setupForDevice(::RAJA::resources::Resource& res)
  {
    bool act = !device && setupReducers();
    if (act) {
      device = res.template allocate<T>(1, ::RAJA::resources::MemoryAccess::Device);
      device_count = static_cast<unsigned int*>(
          res.calloc(sizeof(unsigned int), ::RAJA::resources::MemoryAccess::Device));
      own_device_ptr = true;
    }
    return act;
  }

  //! if own resources teardown device setup
  //  free device pointers
  bool teardownForDevice(::RAJA::resources::Resource& res)
  {
    bool act = own_device_ptr;
    if (act) {
      res.deallocate(device, ::RAJA::resources::MemoryAccess::Device);
      device = nullptr;
      res.deallocate(device_count, ::RAJA::resources::MemoryAccess::Device);
      device_count = nullptr;
      own_device_ptr = false;
    }
    return act;
  }
};

//! Cuda Reduction entity -- generalize on reduction, and type
template <typename Combiner, typename T, bool maybe_atomic>
class Reduce
{
public:
  Reduce() : Reduce(T(), Combiner::identity()) {}

  //! create a reduce object
  //  the original object's parent is itself
  explicit Reduce(T init_val, T identity_ = Combiner::identity())
      : parent{this},
        tally_or_node_ptr{new PinnedTally<T>},
        val(init_val, identity_)
  {
  }

  void reset(T in_val, T identity_ = Combiner::identity())
  {
    operator T();  // syncs device
    val = reduce_data_type(in_val, identity_);
  }

  //! copy and on host attempt to setup for device
  //  init node_ptr to avoid uninitialized read caused by host copy of
  //  reducer in host device lambda not being used on device.
  RAJA_HOST_DEVICE
  Reduce(const Reduce& other)
#if !defined(RAJA_DEVICE_CODE)
      : parent{other.parent},
#else
      : parent{&other},
#endif
        tally_or_node_ptr{other.tally_or_node_ptr},
        val(other.val)
  {
#if !defined(RAJA_DEVICE_CODE)
    if (parent) {
      if (val.setupForDevice(currentResource())) {
        tally_or_node_ptr.node_ptr =
            tally_or_node_ptr.list->new_value(currentCudaResource(), currentResource());
        val.init_grid_val(&tally_or_node_ptr.node_ptr->value);
        parent = nullptr;
      }
    }
#endif
  }

  //! apply reduction upon destruction and cleanup resources owned by this copy
  //  on device store in pinned buffer on host
  RAJA_HOST_DEVICE
  ~Reduce()
  {
#if !defined(RAJA_DEVICE_CODE)
    if (parent == this) {
      delete tally_or_node_ptr.list;
      tally_or_node_ptr.list = nullptr;
    } else if (parent) {
      if (val.value != val.identity) {
#if defined(RAJA_ENABLE_OPENMP)
        lock_guard<omp::mutex> lock(tally_or_node_ptr.list->m_mutex);
#endif
        parent->combine(val.value);
      }
    } else {
      if (val.teardownForDevice(tally_or_node_ptr.node_ptr->res)) {
        tally_or_node_ptr.node_ptr = nullptr;
      }
    }
#else
    if (!parent->parent) {
      val.grid_reduce(&tally_or_node_ptr.node_ptr->value);
    } else {
      parent->combine(val.value);
    }
#endif
  }

  //! map result value back to host if not done already; return aggregate value
  operator T()
  {
    auto n = tally_or_node_ptr.list->begin();
    auto end = tally_or_node_ptr.list->end();
    if (n != end) {
      tally_or_node_ptr.list->synchronize_resources();
      for (; n != end; ++n) {
        Combiner{}(val.value, *n);
      }
      tally_or_node_ptr.list->free_list();
    }
    return val.value;
  }
  //! alias for operator T()
  T get() { return operator T(); }

  //! apply reduction (const version) -- still combines internal values
  RAJA_HOST_DEVICE
  void combine(T other) const { Combiner{}(val.value, other); }

  /*!
   *  \return reference to the local value
   */
  T& local() const { return val.value; }

  T get_combined() const { return val.value; }

private:
  const Reduce* parent;

  //! union to hold either pointer to PinnedTally or poiter to value
  //  only use list before setup for device and only use node_ptr after
  union tally_u {
    PinnedTally<T>* list;
    typename PinnedTally<T>::Node* node_ptr;
    constexpr tally_u(PinnedTally<T>* l) : list(l){};
    constexpr tally_u(typename PinnedTally<T>::Node* n_ptr) : node_ptr(n_ptr){};
  };

  tally_u tally_or_node_ptr;

  //! cuda reduction data storage class and folding algorithm
  using reduce_data_type = typename std::conditional<
      maybe_atomic && RAJA::reduce::cuda::cuda_atomic_available<T>::value,
      cuda::ReduceAtomic_Data<Combiner, T>,
      cuda::Reduce_Data<Combiner, T>>::type;

  //! storage for reduction data
  reduce_data_type val;
};

}  // end namespace cuda

//! specialization of ReduceSum for cuda_reduce
template <bool maybe_atomic, typename T>
class ReduceSum<cuda_reduce_base<maybe_atomic>, T>
    : public cuda::Reduce<RAJA::reduce::sum<T>, T, maybe_atomic>
{

public:
  using Base = cuda::Reduce<RAJA::reduce::sum<T>, T, maybe_atomic>;
  using Base::Base;
  //! enable operator+= for ReduceSum -- alias for combine()
  RAJA_HOST_DEVICE
  const ReduceSum& operator+=(T rhs) const
  {
    this->combine(rhs);
    return *this;
  }
};

//! specialization of ReduceBitOr for cuda_reduce
template <bool maybe_atomic, typename T>
class ReduceBitOr<cuda_reduce_base<maybe_atomic>, T>
    : public cuda::Reduce<RAJA::reduce::or_bit<T>, T, maybe_atomic>
{

public:
  using Base = cuda::Reduce<RAJA::reduce::or_bit<T>, T, maybe_atomic>;
  using Base::Base;
  //! enable operator|= for ReduceBitOr -- alias for combine()
  RAJA_HOST_DEVICE
  const ReduceBitOr& operator|=(T rhs) const
  {
    this->combine(rhs);
    return *this;
  }
};

//! specialization of ReduceBitAnd for cuda_reduce
template <bool maybe_atomic, typename T>
class ReduceBitAnd<cuda_reduce_base<maybe_atomic>, T>
    : public cuda::Reduce<RAJA::reduce::and_bit<T>, T, maybe_atomic>
{

public:
  using Base = cuda::Reduce<RAJA::reduce::and_bit<T>, T, maybe_atomic>;
  using Base::Base;
  //! enable operator&= for ReduceBitAnd -- alias for combine()
  RAJA_HOST_DEVICE
  const ReduceBitAnd& operator&=(T rhs) const
  {
    this->combine(rhs);
    return *this;
  }
};

//! specialization of ReduceMin for cuda_reduce
template <bool maybe_atomic, typename T>
class ReduceMin<cuda_reduce_base<maybe_atomic>, T>
    : public cuda::Reduce<RAJA::reduce::min<T>, T, maybe_atomic>
{

public:
  using Base = cuda::Reduce<RAJA::reduce::min<T>, T, maybe_atomic>;
  using Base::Base;
  //! enable min() for ReduceMin -- alias for combine()
  RAJA_HOST_DEVICE
  const ReduceMin& min(T rhs) const
  {
    this->combine(rhs);
    return *this;
  }
};

//! specialization of ReduceMax for cuda_reduce
template <bool maybe_atomic, typename T>
class ReduceMax<cuda_reduce_base<maybe_atomic>, T>
    : public cuda::Reduce<RAJA::reduce::max<T>, T, maybe_atomic>
{

public:
  using Base = cuda::Reduce<RAJA::reduce::max<T>, T, maybe_atomic>;
  using Base::Base;
  //! enable max() for ReduceMax -- alias for combine()
  RAJA_HOST_DEVICE
  const ReduceMax& max(T rhs) const
  {
    this->combine(rhs);
    return *this;
  }
};

//! specialization of ReduceMinLoc for cuda_reduce
template <bool maybe_atomic, typename T, typename IndexType>
class ReduceMinLoc<cuda_reduce_base<maybe_atomic>, T, IndexType>
    : public cuda::Reduce<RAJA::reduce::min<RAJA::reduce::detail::ValueLoc<T, IndexType>>,
                          RAJA::reduce::detail::ValueLoc<T, IndexType>,
                          maybe_atomic>
{

public:
  using value_type = RAJA::reduce::detail::ValueLoc<T, IndexType>;
  using Base = cuda::
      Reduce<RAJA::reduce::min<value_type>, value_type, maybe_atomic>;
  using Base::Base;

  //! constructor requires a default value for the reducer
  ReduceMinLoc(T init_val, IndexType init_idx)
      : Base(value_type(init_val, init_idx))
  {
  }
  //! reducer function; updates the current instance's state
  RAJA_HOST_DEVICE
  const ReduceMinLoc& minloc(T rhs, IndexType loc) const
  {
    this->combine(value_type(rhs, loc));
    return *this;
  }

  //! Get the calculated reduced value
  IndexType getLoc() { return Base::get().getLoc(); }

  //! Get the calculated reduced value
  operator T() { return Base::get(); }

  //! Get the calculated reduced value
  T get() { return Base::get(); }
};

//! specialization of ReduceMaxLoc for cuda_reduce
template <bool maybe_atomic, typename T, typename IndexType>
class ReduceMaxLoc<cuda_reduce_base<maybe_atomic>, T, IndexType>
    : public cuda::
          Reduce<RAJA::reduce::max<RAJA::reduce::detail::ValueLoc<T, IndexType, false>>,
                 RAJA::reduce::detail::ValueLoc<T, IndexType, false>,
                 maybe_atomic>
{
public:
  using value_type = RAJA::reduce::detail::ValueLoc<T, IndexType, false>;
  using Base = cuda::
      Reduce<RAJA::reduce::max<value_type>, value_type, maybe_atomic>;
  using Base::Base;

  //! constructor requires a default value for the reducer
  ReduceMaxLoc(T init_val, IndexType init_idx)
      : Base(value_type(init_val, init_idx))
  {
  }
  //! reducer function; updates the current instance's state
  RAJA_HOST_DEVICE
  const ReduceMaxLoc& maxloc(T rhs, IndexType loc) const
  {
    this->combine(value_type(rhs, loc));
    return *this;
  }

  //! Get the calculated reduced value
  IndexType getLoc() { return Base::get().getLoc(); }

  //! Get the calculated reduced value
  operator T() { return Base::get(); }

  //! Get the calculated reduced value
  T get() { return Base::get(); }
};

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
