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

#ifndef RAJA_cuda_reduce_HPP
#define RAJA_cuda_reduce_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "RAJA/util/types.hpp"

#include "RAJA/util/basic_mempool.hpp"

#include "RAJA/util/SoAArray.hpp"

#include "RAJA/util/SoAPtr.hpp"

#include "RAJA/util/mutex.hpp"

#include "RAJA/pattern/detail/reduce.hpp"

#include "RAJA/pattern/reduce.hpp"

#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"

#include "RAJA/policy/cuda/policy.hpp"

#include "RAJA/policy/cuda/atomic.hpp"

#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"

#include <cuda.h>

#include <type_traits>

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
    RAJA::atomic::atomicAdd<T>(RAJA::atomic::cuda_atomic{}, &val, v);
  }
};

template <typename T>
struct atomic<min<T>> {
  RAJA_DEVICE RAJA_INLINE void operator()(T& val, const T v)
  {
    RAJA::atomic::atomicMin<T>(RAJA::atomic::cuda_atomic{}, &val, v);
  }
};

template <typename T>
struct atomic<max<T>> {
  RAJA_DEVICE RAJA_INLINE void operator()(T& val, const T v)
  {
    RAJA::atomic::atomicMax<T>(RAJA::atomic::cuda_atomic{}, &val, v);
  }
};

template <typename T>
struct cuda_atomic_available {
  static constexpr const bool value =
      (std::is_integral<T>::value && (4 == sizeof(T) || 8 == sizeof(T)))
      || std::is_same<T, float>::value || std::is_same<T, double>::value;
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
  using integer_type = typename std::
      conditional<((alignof(T) >= alignof(long long)
                    && sizeof(long long) <= max_integer_type_size)
                   || sizeof(long) < min_integer_type_size),
                  long long,
                  typename std::
                      conditional<((alignof(T) >= alignof(long)
                                    && sizeof(long) <= max_integer_type_size)
                                   || sizeof(int) < min_integer_type_size),
                                  long,
                                  typename std::
                                      conditional<((alignof(T) >= alignof(int)
                                                    && sizeof(int)
                                                           <= max_integer_type_size)
                                                   || sizeof(short)
                                                          < min_integer_type_size),
                                                  int,
                                                  typename std::
                                                      conditional<((alignof(T)
                                                                        >= alignof(
                                                                               short)
                                                                    && sizeof(
                                                                           short)
                                                                           <= max_integer_type_size)
                                                                   || sizeof(
                                                                          char)
                                                                          < min_integer_type_size),
                                                                  short,
                                                                  typename std::
                                                                      conditional<((alignof(
                                                                                        T)
                                                                                        >= alignof(
                                                                                               char)
                                                                                    && sizeof(
                                                                                           char)
                                                                                           <= max_integer_type_size)),
                                                                                  char,
                                                                                  void>::
                                                                          type>::
                                                          type>::type>::type>::
          type;
  static_assert(!std::is_same<integer_type, void>::value,
                "could not find a compatible integer type");
  static_assert(sizeof(integer_type) >= min_integer_type_size,
                "integer_type smaller than min integer type size");
  static_assert(sizeof(integer_type) <= max_integer_type_size,
                "integer_type greater than max integer type size");

  constexpr static size_t num_integer_type =
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
#if (__CUDACC_VER_MAJOR__ >= 9)
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

  for (int i = 0; i < u.array_size(); ++i) {
#if (__CUDACC_VER_MAJOR__ >= 9)
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

  for (int i = 0; i < u.array_size(); ++i) {
#if (__CUDACC_VER_MAJOR__ >= 9)
    u.array[i] = ::__shfl_sync(0xffffffffu, u.array[i], srcLane);
#else
    u.array[i] = ::__shfl(u.array[i], srcLane);
#endif
  }
  return u.value;
}

//! reduce values in block into thread 0
template <typename Combiner, typename T>
RAJA_DEVICE RAJA_INLINE T block_reduce(T val, T identity)
{
  int numThreads = blockDim.x * blockDim.y * blockDim.z;

  int threadId = threadIdx.x + blockDim.x * threadIdx.y
                 + (blockDim.x * blockDim.y) * threadIdx.z;

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

    __shared__ RAJA::detail::SoAArray<T, policy::cuda::MAX_WARPS> sd;

    // write per warp values to shared memory
    if (warpId == 0) {
      sd.set(warpNum, temp);
    }

    __syncthreads();

    if (warpNum == 0) {

      // read per warp values
      if (warpId * policy::cuda::WARP_SIZE < numThreads) {
        temp = sd.get(warpId);
      } else {
        temp = identity;
      }

      for (int i = 1; i < policy::cuda::WARP_SIZE; i *= 2) {
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

  int blockId = blockIdx.x + gridDim.x * blockIdx.y
                + (gridDim.x * gridDim.y) * blockIdx.z;

  int threadId = threadIdx.x + blockDim.x * threadIdx.y
                 + (blockDim.x * blockDim.y) * threadIdx.z;

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

  int threadId = threadIdx.x + blockDim.x * threadIdx.y
                 + (blockDim.x * blockDim.y) * threadIdx.z;

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
    T value;
  };
  //! Object per stream to keep track of pinned memory nodes
  struct StreamNode {
    StreamNode* next;
    cudaStream_t stream;
    Node* node_list;
  };

  //! Iterator over streams used by reducer
  class StreamIterator
  {
  public:
    StreamIterator() = delete;

    StreamIterator(StreamNode* sn) : m_sn(sn) {}

    const StreamIterator& operator++()
    {
      m_sn = m_sn->next;
      return *this;
    }

    StreamIterator operator++(int)
    {
      StreamIterator ret = *this;
      this->operator++();
      return ret;
    }

    cudaStream_t& operator*() { return m_sn->stream; }

    bool operator==(const StreamIterator& rhs) const
    {
      return m_sn == rhs.m_sn;
    }

    bool operator!=(const StreamIterator& rhs) const
    {
      return !this->operator==(rhs);
    }

  private:
    StreamNode* m_sn;
  };

  //! Iterator over all values generated by reducer
  class StreamNodeIterator
  {
  public:
    StreamNodeIterator() = delete;

    StreamNodeIterator(StreamNode* sn, Node* n) : m_sn(sn), m_n(n) {}

    const StreamNodeIterator& operator++()
    {
      if (m_n->next) {
        m_n = m_n->next;
      } else if (m_sn->next) {
        m_sn = m_sn->next;
        m_n = m_sn->node_list;
      } else {
        m_sn = nullptr;
        m_n = nullptr;
      }
      return *this;
    }

    StreamNodeIterator operator++(int)
    {
      StreamNodeIterator ret = *this;
      this->operator++();
      return ret;
    }

    T& operator*() { return m_n->value; }

    bool operator==(const StreamNodeIterator& rhs) const
    {
      return m_n == rhs.m_n;
    }

    bool operator!=(const StreamNodeIterator& rhs) const
    {
      return !this->operator==(rhs);
    }

  private:
    StreamNode* m_sn;
    Node* m_n;
  };

  PinnedTally() : stream_list(nullptr) {}

  PinnedTally(const PinnedTally&) = delete;

  //! get begin iterator over streams
  StreamIterator streamBegin() { return {stream_list}; }

  //! get end iterator over streams
  StreamIterator streamEnd() { return {nullptr}; }

  //! get begin iterator over values
  StreamNodeIterator begin()
  {
    return {stream_list, stream_list ? stream_list->node_list : nullptr};
  }

  //! get end iterator over values
  StreamNodeIterator end() { return {nullptr, nullptr}; }

  //! get new value for use in stream
  T* new_value(cudaStream_t stream)
  {
#if defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)
    lock_guard<omp::mutex> lock(m_mutex);
#endif
    StreamNode* sn = stream_list;
    while (sn) {
      if (sn->stream == stream) break;
      sn = sn->next;
    }
    if (!sn) {
      sn = (StreamNode*)malloc(sizeof(StreamNode));
      sn->next = stream_list;
      sn->stream = stream;
      sn->node_list = nullptr;
      stream_list = sn;
    }
    Node* n = cuda::pinned_mempool_type::getInstance().template malloc<Node>(1);
    n->next = sn->node_list;
    sn->node_list = n;
    return &n->value;
  }

  //! synchronize all streams used
  void synchronize_streams()
  {
    auto end = streamEnd();
    for (auto s = streamBegin(); s != end; ++s) {
      synchronize(*s);
    }
  }

  //! all values used in all streams
  void free_list()
  {
    while (stream_list) {
      StreamNode* s = stream_list;
      while (s->node_list) {
        Node* n = s->node_list;
        s->node_list = n->next;
        cuda::pinned_mempool_type::getInstance().free(n);
      }
      stream_list = s->next;
      free(s);
    }
  }

  ~PinnedTally() { free_list(); }

#if defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)
  omp::mutex m_mutex;
#endif

private:
  StreamNode* stream_list;
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
template <bool Async, typename Combiner, typename T>
struct Reduce_Data {

  mutable T value;
  T identity;
  unsigned int* device_count;
  RAJA::detail::SoAPtr<T, device_mempool_type> device;
  bool own_device_ptr;

  Reduce_Data() { reset(T(), T()); }

  /*! \brief create from a default value and offload information
   *
   *  allocates PinnedTally to hold device values
   */
  Reduce_Data(T initValue, T identity_) { reset(initValue, identity_); }

  void reset(T initValue, T identity_ = T())
  {
    cudaDeviceSynchronize();
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
  bool setupForDevice()
  {
    bool act = !device.allocated() && setupReducers();
    if (act) {
      dim3 gridDim = currentGridDim();
      size_t numBlocks = gridDim.x * gridDim.y * gridDim.z;
      device.allocate(numBlocks);
      device_count = device_zeroed_mempool_type::getInstance()
                         .template malloc<unsigned int>(1);
      own_device_ptr = true;
    }
    return act;
  }

  //! if own resources teardown device setup
  //  free device pointers
  bool teardownForDevice()
  {
    bool act = own_device_ptr;
    if (act) {
      device.deallocate();
      device_zeroed_mempool_type::getInstance().free(device_count);
      device_count = nullptr;
      own_device_ptr = false;
    }
    return act;
  }
};


//! Reduction data for Cuda Offload -- stores value, host pointer
template <bool Async, typename Combiner, typename T>
struct ReduceAtomic_Data {

  mutable T value;
  T identity;
  unsigned int* device_count;
  T* device;
  bool own_device_ptr;

  ReduceAtomic_Data() { reset(T(), T()); }

  ReduceAtomic_Data(T initValue, T identity_) { reset(initValue, identity_); }

  void reset(T initValue, T identity_ = Combiner::identity())
  {
    cudaDeviceSynchronize();
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
  bool setupForDevice()
  {
    bool act = !device && setupReducers();
    if (act) {
      device = device_mempool_type::getInstance().template malloc<T>(1);
      device_count = device_zeroed_mempool_type::getInstance()
                         .template malloc<unsigned int>(1);
      own_device_ptr = true;
    }
    return act;
  }

  //! if own resources teardown device setup
  //  free device pointers
  bool teardownForDevice()
  {
    bool act = own_device_ptr;
    if (act) {
      device_mempool_type::getInstance().free(device);
      device = nullptr;
      device_zeroed_mempool_type::getInstance().free(device_count);
      device_count = nullptr;
      own_device_ptr = false;
    }
    return act;
  }
};

//! Cuda Reduction entity -- generalize on reduction, and type
template <bool Async, typename Combiner, typename T, bool maybe_atomic>
class Reduce
{
public:
  Reduce() : parent{this}, tally_or_val_ptr{new PinnedTally<T>}
  {
    reset(T(), Combiner::identity());
  }

  //! create a reduce object
  //  the original object's parent is itself
  explicit Reduce(T init_val, T identity_ = Combiner::identity())
      : parent{this}, tally_or_val_ptr{new PinnedTally<T>}
  {
    reset(init_val, identity_);
  }

  void reset(T in_val, T identity_ = Combiner::identity())
  {
    cudaDeviceSynchronize();
    val = reduce_data_type(in_val, identity_);
  }

  //! copy and on host attempt to setup for device
  RAJA_HOST_DEVICE
  Reduce(const Reduce& other)
#if !defined(__CUDA_ARCH__)
      : parent{other.parent},
#else
      : parent{&other},
#endif
        tally_or_val_ptr{other.tally_or_val_ptr},
        val(other.val)
  {
#if !defined(__CUDA_ARCH__)
    if (parent) {
      if (val.setupForDevice()) {
        tally_or_val_ptr.val_ptr =
            tally_or_val_ptr.list->new_value(currentStream());
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
#if !defined(__CUDA_ARCH__)
    if (parent == this) {
      delete tally_or_val_ptr.list;
      tally_or_val_ptr.list = nullptr;
    } else if (parent) {
      if (val.value != val.identity) {
#if defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)
        lock_guard<omp::mutex> lock(tally_or_val_ptr.list->m_mutex);
#endif
        parent->combine(val.value);
      }
    } else {
      if (val.teardownForDevice()) {
        tally_or_val_ptr.val_ptr = nullptr;
      }
    }
#else
    if (!parent->parent) {
      val.grid_reduce(tally_or_val_ptr.val_ptr);
    } else {
      parent->combine(val.value);
    }
#endif
  }

  //! map result value back to host if not done already; return aggregate value
  operator T()
  {
    auto n = tally_or_val_ptr.list->begin();
    auto end = tally_or_val_ptr.list->end();
    if (n != end) {
      tally_or_val_ptr.list->synchronize_streams();
      for (; n != end; ++n) {
        Combiner{}(val.value, *n);
      }
      tally_or_val_ptr.list->free_list();
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
  //  only use list before setup for device and only use val_ptr after
  union tally_u {
    PinnedTally<T>* list;
    T* val_ptr;
    constexpr tally_u(PinnedTally<T>* l) : list(l){};
    constexpr tally_u(T* v_ptr) : val_ptr(v_ptr){};
  };

  tally_u tally_or_val_ptr;

  //! cuda reduction data storage class and folding algorithm
  using reduce_data_type = typename std::
      conditional<maybe_atomic
                      && RAJA::reduce::cuda::cuda_atomic_available<T>::value,
                  cuda::ReduceAtomic_Data<Async, Combiner, T>,
                  cuda::Reduce_Data<Async, Combiner, T>>::type;

  //! storage for reduction data
  reduce_data_type val;
};

}  // end namespace cuda

//! specialization of ReduceSum for cuda_reduce
template <size_t BLOCK_SIZE, bool Async, bool maybe_atomic, typename T>
class ReduceSum<cuda_reduce<BLOCK_SIZE, Async, maybe_atomic>, T>
    : public cuda::Reduce<Async, RAJA::reduce::sum<T>, T, maybe_atomic>
{

public:
  using Base = cuda::Reduce<Async, RAJA::reduce::sum<T>, T, maybe_atomic>;
  using Base::Base;
  //! enable operator+= for ReduceSum -- alias for combine()
  RAJA_HOST_DEVICE
  const ReduceSum& operator+=(T rhs) const
  {
    this->combine(rhs);
    return *this;
  }
};

//! specialization of ReduceMin for cuda_reduce
template <size_t BLOCK_SIZE, bool Async, bool maybe_atomic, typename T>
class ReduceMin<cuda_reduce<BLOCK_SIZE, Async, maybe_atomic>, T>
    : public cuda::Reduce<Async, RAJA::reduce::min<T>, T, maybe_atomic>
{

public:
  using Base = cuda::Reduce<Async, RAJA::reduce::min<T>, T, maybe_atomic>;
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
template <size_t BLOCK_SIZE, bool Async, bool maybe_atomic, typename T>
class ReduceMax<cuda_reduce<BLOCK_SIZE, Async, maybe_atomic>, T>
    : public cuda::Reduce<Async, RAJA::reduce::max<T>, T, maybe_atomic>
{

public:
  using Base = cuda::Reduce<Async, RAJA::reduce::max<T>, T, maybe_atomic>;
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
template <size_t BLOCK_SIZE, bool Async, bool maybe_atomic, typename T>
class ReduceMinLoc<cuda_reduce<BLOCK_SIZE, Async, maybe_atomic>, T>
    : public cuda::Reduce<Async,
                          RAJA::reduce::min<RAJA::reduce::detail::ValueLoc<T>>,
                          RAJA::reduce::detail::ValueLoc<T>,
                          maybe_atomic>
{

public:
  using value_type = RAJA::reduce::detail::ValueLoc<T>;
  using Base = cuda::
      Reduce<Async, RAJA::reduce::min<value_type>, value_type, maybe_atomic>;
  using Base::Base;

  //! constructor requires a default value for the reducer
  explicit ReduceMinLoc(T init_val, Index_type init_idx)
      : Base(value_type(init_val, init_idx))
  {
  }
  //! reducer function; updates the current instance's state
  RAJA_HOST_DEVICE
  const ReduceMinLoc& minloc(T rhs, Index_type loc) const
  {
    this->combine(value_type(rhs, loc));
    return *this;
  }

  //! Get the calculated reduced value
  Index_type getLoc() { return Base::get().getLoc(); }

  //! Get the calculated reduced value
  operator T() { return Base::get(); }

  //! Get the calculated reduced value
  T get() { return Base::get(); }
};

//! specialization of ReduceMaxLoc for cuda_reduce
template <size_t BLOCK_SIZE, bool Async, bool maybe_atomic, typename T>
class ReduceMaxLoc<cuda_reduce<BLOCK_SIZE, Async, maybe_atomic>, T>
    : public cuda::
          Reduce<Async,
                 RAJA::reduce::max<RAJA::reduce::detail::ValueLoc<T, false>>,
                 RAJA::reduce::detail::ValueLoc<T, false>,
                 maybe_atomic>
{
public:
  using value_type = RAJA::reduce::detail::ValueLoc<T, false>;
  using Base = cuda::
      Reduce<Async, RAJA::reduce::max<value_type>, value_type, maybe_atomic>;
  using Base::Base;

  //! constructor requires a default value for the reducer
  explicit ReduceMaxLoc(T init_val, Index_type init_idx)
      : Base(value_type(init_val, init_idx))
  {
  }
  //! reducer function; updates the current instance's state
  RAJA_HOST_DEVICE
  const ReduceMaxLoc& maxloc(T rhs, Index_type loc) const
  {
    this->combine(value_type(rhs, loc));
    return *this;
  }

  //! Get the calculated reduced value
  Index_type getLoc() { return Base::get().getLoc(); }

  //! Get the calculated reduced value
  operator T() { return Base::get(); }

  //! Get the calculated reduced value
  T get() { return Base::get(); }
};

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
