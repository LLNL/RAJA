/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA reduction templates for ROCM execution.
 *
 *          These methods should work on any platform that supports
 *          ROCM devices.
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

#ifndef RAJA_rocm_reduce_HPP
#define RAJA_rocm_reduce_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_ROCM)

#include "RAJA/util/types.hpp"

#include "RAJA/util/basic_mempool.hpp"

#include "RAJA/util/SoAArray.hpp"

#include "RAJA/util/SoAPtr.hpp"

#include "RAJA/util/mutex.hpp"

#include "RAJA/pattern/detail/reduce.hpp"

#include "RAJA/pattern/reduce.hpp"

#include "RAJA/policy/rocm/MemUtils_ROCm.hpp"

#include "RAJA/policy/rocm/policy.hpp"

#include "RAJA/policy/rocm/atomic.hpp"

#include "RAJA/policy/rocm/raja_rocmerrchk.hpp"


#include <type_traits>

namespace RAJA
{

namespace reduce
{

namespace rocm
{
//! atomic operator version of Combiner object
template <typename Combiner>
struct atomic;

template <typename T>
struct atomic<sum<T>> {
  RAJA_DEVICE RAJA_INLINE void operator()(T& val, const T v)
  {
//    RAJA::atomic::atomicAdd<T>(RAJA::atomic::rocm_atomic{}, &val, v);
    RAJA::atomic::atomicAdd<T>(&val, v);
  }
};

template <typename T>
struct atomic<min<T>> {
  RAJA_DEVICE RAJA_INLINE void operator()(T& val, const T v)
  {
//    RAJA::atomic::atomicMin<T>(RAJA::atomic::rocm_atomic{}, &val, v);
    RAJA::atomic::atomicMin<T>(&val, v);
  }
};

template <typename T>
struct atomic<max<T>> {
  RAJA_DEVICE RAJA_INLINE void operator()(T& val, const T v)
  {
//    RAJA::atomic::atomicMax<T>(RAJA::atomic::rocm_atomic{}, &val, v);
    RAJA::atomic::atomicMax<T>(&val, v);
  }
};

template <typename T>
struct rocm_atomic_available {
  static constexpr const bool value =
      (std::is_integral<T>::value && (4 == sizeof(T) || 8 == sizeof(T)))
      || std::is_same<T, float>::value || std::is_same<T, double>::value;
};

}  // namespace rocm

}  // namespace reduce

namespace rocm
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

// rocm only has shfl primitives for 32 bits 
constexpr const size_t min_shfl_int_type_size = sizeof(int);
constexpr const size_t max_shfl_int_type_size = sizeof(int);

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
#if __KALMAR_ACCELERATOR__ == 1
template <typename T>
RAJA_DEVICE RAJA_INLINE T shfl_xor_sync(T var, int laneMask)
{
  AsIntegerArray<T, min_shfl_int_type_size, max_shfl_int_type_size> u(var);

  for (int i = 0; i < u.array_size(); ++i) {
//#if (__ROCMCC_VER_MAJOR__ >= 9)
//    u.array[i] = ::__shfl_xor_sync(0xffffffffu, u.array[i], laneMask);
//#else
    u.array[i] = hc::__shfl_xor(u.array[i], laneMask);
//#endif
  }
  return u.value;
}

template <typename T>
RAJA_DEVICE RAJA_INLINE T shfl_sync(T var, int srcLane)
{
  AsIntegerArray<T, min_shfl_int_type_size, max_shfl_int_type_size> u(var);

  for (int i = 0; i < u.array_size(); ++i) {
//#if (__ROCMCC_VER_MAJOR__ >= 9)
//    u.array[i] = ::__shfl_sync(0xffffffffu, u.array[i], srcLane);
//#else
    u.array[i] = hc::__shfl(u.array[i], srcLane);
//#endif
  }
  return u.value;
}
/*
RAJA_DEVICE RAJA_INLINE
void __syncthreads()
{
#if __KALMAR_ACCELERATOR__ == 1
   amp_barrier(CLK_LOCAL_MEM_FENCE);
#else
#endif
}
RAJA_DEVICE RAJA_INLINE
void __threadfence()
{
#if __KALMAR_ACCELERATOR__ == 1
   amp_barrier(CLK_GLOBAL_MEM_FENCE);
#else
#endif
}
*/
//! reduce values in block into thread 0
template <typename Combiner, typename T>
RAJA_DEVICE RAJA_INLINE T block_reduce(T val, T identity) [[hc]]
{
  int numThreads = blockDim_x * blockDim_y * blockDim_z;

  int threadId = threadIdx_x + blockDim_x * threadIdx_y
                 + (blockDim_x * blockDim_y) * threadIdx_z;

  int wfId = threadId % policy::rocm::WAVEFRONT_SIZE;
  int wfNum = threadId / policy::rocm::WAVEFRONT_SIZE;

  T temp = val;

  if (numThreads % policy::rocm::WAVEFRONT_SIZE == 0) {

    // reduce each wf
    for (int i = 1; i < policy::rocm::WAVEFRONT_SIZE; i *= 2) {
      T rhs = shfl_xor_sync(temp, i);
      Combiner{}(temp, rhs);
    }

  } else {

    // reduce each wf
    for (int i = 1; i < policy::rocm::WAVEFRONT_SIZE; i *= 2) {
      int srcLane = threadId ^ i;
      T rhs = shfl_sync(temp, srcLane);
      // only add from threads that exist (don't double count own value)
      if (srcLane < numThreads) {
        Combiner{}(temp, rhs);
      }
    }
  }

  // reduce per wf values
  if (numThreads > policy::rocm::WAVEFRONT_SIZE) {

    tile_static RAJA::detail::SoAArray<T, policy::rocm::MAX_WAVEFRONTS> sd;

    // write per wf values to shared memory
    if (wfId == 0) {
      sd.set(wfNum, temp);
    }

    __syncthreads();

    if (wfNum == 0) {

      // read per wf values
      if (wfId * policy::rocm::WAVEFRONT_SIZE < numThreads) {
        temp = sd.get(wfId);
      } else {
        temp = identity;
      }

      for (int i = 1; i < policy::rocm::WAVEFRONT_SIZE; i *= 2) {
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
                                         unsigned int* device_count) [[hc]]
{
  int numBlocks = gridDim_x * gridDim_y * gridDim_z;
  int numThreads = blockDim_x * blockDim_y * blockDim_z;
  unsigned int wrap_around = numBlocks - 1;

  int blockId = blockIdx.x + gridDim_x * blockIdx.y
                + (gridDim_x * gridDim_y) * blockIdx.z;

  int threadId = threadIdx_x + blockDim_x * threadIdx_y
                 + (blockDim_x * blockDim_y) * threadIdx_z;

  T temp = block_reduce<Combiner>(val, identity);

  // one thread per block writes to device_mem
  bool lastBlock = false;
  if (threadId == 0) {
    device_mem.set(blockId, temp);
    // ensure write visible to all threadblocks
    __threadfence();
    // increment counter, (wraps back to zero if old count == wrap_around)
    unsigned int old_count = RAJA::atomic::atomicInc(RAJA::atomic::rocm_atomic{},device_count, wrap_around);
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
val = (T)5;
  return lastBlock && threadId == 0;
}


//! reduce values in grid into thread 0 of last running block
//  returns true if put reduced value in val
template <typename Combiner, typename T>
RAJA_DEVICE RAJA_INLINE bool grid_reduce_atomic(T& val,
                                                T identity,
                                                T* device_mem,
                                                unsigned int* device_count)
                                                [[hc]]
{
  int numBlocks = gridDim_x * gridDim_y * gridDim_z;
  unsigned int wrap_around = numBlocks + 1;

  int threadId = threadIdx_x + blockDim_x * threadIdx_y
                 + (blockDim_x * blockDim_y) * threadIdx_z;

  // one thread in first block initializes device_mem
  if (threadId == 0) {
    unsigned int old_val = RAJA::atomic::atomicCAS(RAJA::atomic::rocm_atomic{},device_count, 0u, 1u);
    if (old_val == 0u) {
      device_mem[0] = identity;
      __threadfence();
      RAJA::atomic::atomicAdd(RAJA::atomic::rocm_atomic{},device_count, 1u);
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
    RAJA::reduce::rocm::atomic<Combiner>{}(device_mem[0], temp);
    __threadfence();
    // increment counter, (wraps back to zero if old count == wrap_around)
    unsigned int old_count = RAJA::atomic::atomicInc(RAJA::atomic::rocm_atomic{},device_count, wrap_around);
    lastBlock = (old_count == wrap_around);

    // last block gets value from device_mem
    if (lastBlock) {
      val = device_mem[0];
    }
  }

  return lastBlock;
}
#endif

}  // namespace impl
#if 0

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
    rocmStream_t stream;
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

    rocmStream_t& operator*() { return m_sn->stream; }

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
  T* new_value(rocmStream_t stream)
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
    Node* n = rocm::pinned_mempool_type::getInstance().template malloc<Node>(1);
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
        rocm::pinned_mempool_type::getInstance().free(n);
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
#endif

//
//////////////////////////////////////////////////////////////////////
//
// Reduction classes.
//
//////////////////////////////////////////////////////////////////////
//
// TODO:  clean this up.  The if 0 tries to mimic the cuda way.
//    it is not working yet
// the else is the previous rocm reductions (pre github) modified for
//    discrete GPUs.  It passes ex2 right now.
#if 0
//! Reduction data for ROCm Offload -- stores value, host pointer, and device
//! pointer
template <bool Async, typename Combiner, typename T>
struct Reduce_Data {

  mutable T value;
  T identity;
  unsigned int* device_count;
  RAJA::detail::SoAPtr<T, device_mempool_type> device;
  bool own_device_ptr;

  //! disallow default constructor
  Reduce_Data() = delete;

  /*! \brief create from a default value and offload information
   *
   *  allocates PinnedTally to hold device values
   */
  explicit Reduce_Data(T initValue, T identity_)
      : value{initValue},
        identity{identity_},
        device_count{nullptr},
        device{},
        own_device_ptr{false}
  {
      printf("initValue= %d identity_ = %d\n",(int) initValue,identity_);
//value = (T) 22;
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

#if __KALMAR_ACCELERATOR__ == 1
  RAJA_DEVICE
  void grid_reduce(T* output)
  {
    T temp = value;

    if (impl::grid_reduce<Combiner>(temp, identity, device, device_count)) {
      *output = temp;
    }
*output = (T) 7;
  }
#endif

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
      printf("numBlocks = %d\n",(int)numBlocks);
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


//! Reduction data for ROCm Offload -- stores value, host pointer
template <bool Async, typename Combiner, typename T>
struct ReduceAtomic_Data {

  mutable T value;
  T identity;
  unsigned int* device_count;
  T* device;
  bool own_device_ptr;

  //! disallow default constructor
  ReduceAtomic_Data() = delete;

  /*! \brief create from a default value and offload information
   *
   *  allocates PinnedTally to hold device values
   */
  explicit ReduceAtomic_Data(T initValue, T identity_)
      : value{initValue},
        identity{identity_},
        device_count{nullptr},
        device{nullptr},
        own_device_ptr{false}
  {
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

#if __KALMAR_ACCELERATOR__ == 1
  RAJA_DEVICE
  void grid_reduce(T* output)
  {
    T temp = value;

    if (impl::grid_reduce_atomic<Combiner>(
            temp, identity, device, device_count)) {
      *output = temp;
    }
  }
#endif

  //! check and setup for device
  //  allocate device pointers and get a new result buffer from the pinned tally
  bool setupForDevice()
  {
    bool act = !device && setupReducers();
//    bool act = !device ;
    if (act) {
      device = device_mempool_type::getInstance().template malloc<T>(1);
      device_count = device_zeroed_mempool_type::getInstance()
                         .template malloc<unsigned int>(1);
      own_device_ptr = true;
    }
     // *device = (T)1.0;
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

//! ROCm Reduction entity -- generalize on reduction, and type
template <bool Async, typename Combiner, typename T, bool maybe_atomic>
class Reduce
{
public:
  Reduce() = delete;

  //! create a reduce object
  //  the original object's parent is itself
  explicit Reduce(T init_val, T identity_ = Combiner::identity())
      : parent{this},
        tally_or_val_ptr{new PinnedTally<T>},
        val(init_val, identity_)
  {
printf("reduce\n");
      printf("init_val= %d identity_ = %d\n",(int) init_val,identity_);
//val.value = (T) 2; // works as expected
  }

  //! copy and on host attempt to setup for device
  RAJA_HOST_DEVICE
  Reduce(const Reduce& other)
//#if !defined(__ROCM_ARCH__)
#if __KALMAR_ACCELERATOR__ == 1
      : parent{&other},
#else
      : parent{other.parent},
#endif
        tally_or_val_ptr{other.tally_or_val_ptr},
        val(other.val)
  {
//#if !defined(__ROCM_ARCH__)
#if __KALMAR_ACCELERATOR__ != 1
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
//#if !defined(__ROCM_ARCH__)
#if __KALMAR_ACCELERATOR__ != 1
printf("~Reduce\n");
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
//        Combiner{}(val.value, 5);
      }
      tally_or_val_ptr.list->free_list();
    }
    return val.value;
    return (T) 1;
  }
  //! alias for operator T()
  T get() { return operator T(); }

  //! apply reduction (const version) -- still combines internal values
  RAJA_HOST_DEVICE
//  void combine(T other) const { Combiner{}(val.value, other); }
  void combine(T other) const { Combiner{}(val.value, 3); }

private:
  const Reduce* parent;

  //! union to hold either pointer to PinnedTally or pointer to value
  //  only use list before setup for device and only use val_ptr after
  union tally_u {
    PinnedTally<T>* list;
    T* val_ptr;
    constexpr tally_u(PinnedTally<T>* l) : list(l){};
    constexpr tally_u(T* v_ptr) : val_ptr(v_ptr){};
  };

  tally_u tally_or_val_ptr;

  //! rocm reduction data storage class and folding algorithm
  using reduce_data_type = typename std::
      conditional<maybe_atomic
                      && RAJA::reduce::rocm::rocm_atomic_available<T>::value,
                  rocm::ReduceAtomic_Data<Async, Combiner, T>,
                  rocm::Reduce_Data<Async, Combiner, T>>::type;

  //! storage for reduction data
  reduce_data_type val;
};
#else
static const std::size_t tile_size = 256;
static const std::size_t segment_size = 32768;
static const std::size_t max_reductions = 4;
static const std::size_t block_size = segment_size / max_reductions;
static const std::size_t max_size = block_size / tile_size;

struct assert_check
{
    mutable bool success;
    RAJA_INLINE assert_check() : success(true)
    {}

    RAJA_INLINE void operator()(bool cond) const
    {
        if (!cond) success = false;
    }

    ~assert_check()
    {
        if (not success)
        {
            std::cout << "Assertion check failed" << std::endl;
            std::abort();
        }
    }
};
template<class T>
struct copy_ptr
{
   bool is_copy;
   T* data;

   template<class... Ts>
   RAJA_INLINE copy_ptr(Ts... xs) [[cpu]] [[hc]]
   : is_copy(false), data(new T(xs...))
   {}

   RAJA_INLINE copy_ptr(const copy_ptr& rhs)
   : is_copy(true), data(rhs.data)
   {}

   RAJA_INLINE copy_ptr& operator=(copy_ptr rhs)
   {
      std::swap(this->b, rhs.b);
      std::swap(this->data, rhs.data);
      return *this;
   }

   T& operator*() const
   {
      return *data;
   }

   T* operator->() const
   {
      return data;
   }

   // Cleanup should only happen on the cpu
   void cleanup() [[cpu]]
   {
      if (not is_copy) delete data;
      data = nullptr;
   }

   void cleanup() [[hc]]
   {
      data = nullptr;
   }

   ~copy_ptr()
   {
      this->cleanup();
   }
};

RAJA_INLINE void abort_msg(const char* message)
{
   printf("%s\n", message);
   std::abort();
}
RAJA_INLINE void barrier() [[hc]]
{
   amp_barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);;
}

RAJA_INLINE void barrier() [[cpu]]
{
   abort_msg("Call to to function hcc::barrier() invalid on cpu");
}
#if 0
RAJA_INLINE void* get_group_memory(unsigned offset) [[hc]]
{
   char * ptr = (char *)hc::get_group_segment_base_pointer();
   return (void *)(ptr+offset);
}

RAJA_INLINE  void* get_group_memory(unsigned offset) [[cpu]]
{
   abort_msg("Call to to function hcc::get_group_memory() invalid on cpu");
   return nullptr;
}

#endif
struct reductions
{
   struct entry
   {
      std::function<void(int)> setup;
   };

   static std::array<entry, max_reductions> reduction_table;

   template<class F>
   RAJA_INLINE static void for_each(F f)
   {
      for(auto&& x:reduction_table)
      {
         if (x.setup != nullptr) f(x);
      }
   }

   template<class Reducer>
   RAJA_INLINE static int add(Reducer& r)
   {
      for(int i=0;i<reduction_table.size();i++)
      {
         auto&& x = reduction_table[i];
         if (x.setup == nullptr)
         {
            x.setup = [&](int tile_len) { r.setup(tile_len); };
            return i;
         }
      }
      return -1;
   }

   RAJA_INLINE static void remove(int i)
   {
      auto&& x = reduction_table[i];
      x.setup = nullptr;
   }
};

std::array<reductions::entry, max_reductions> reductions::reduction_table = {};

template<class T, class Joiner>
struct reduce_data
{
   T value;
//   T* result;
//   int length;
//   int tiles;
   int id;
 

   RAJA_INLINE reduce_data(T init_val)
   : value(init_val), id(reductions::add(*this))
   {}

//   RAJA_INLINE
//   T get_default_value() const
//   {
//      return Joiner::identity();
//   }

   void setup(int tile_len)
   { }

   ~reduce_data() 
   {
      reductions::remove(this->id);
   }
};


template<class Joiner, typename T>
class reducer
{
public:
   static_assert(std::is_trivial<T>::value, "Reductions only supported for trivial types");


   copy_ptr<reduce_data<T, Joiner>> data;


   RAJA_INLINE RAJA_HOST_DEVICE reducer(T init_val) 
   : data{init_val}
   { }

///  ~reducer


   RAJA_INLINE T final_result() const
   {
//      return std::accumulate(data->result.begin(), data->result.end(), data->value, Joiner());
    typedef  struct RAJA::rocm::detail::rocmInfo RI;
    RI & rocm_info = RAJA::rocm::detail::tl_status;
    T sum = Joiner::identity();
    int tiles = rocm_info.tiles;
    // copy just the result buffers back to the host.
    rocmMemcpy(rocm_info.device_mem_ptr,rocm_info.host_mem_ptr,tiles*sizeof(T));
    for(int i=0;i<tiles;i++) sum += ((T *) rocm_info.host_mem_ptr)[i];
    free(rocm_info.host_mem_ptr);
    return sum;
   }

   operator T()
   {
      return (this->final_result());
   }

   RAJA_INLINE
   T* local_mem() const [[hc]]
   {
      return (T *) get_group_memory(block_size * data->id);
   }

   template<class X, class Y>
   RAJA_INLINE static void join(X& x, const Y& y) [[cpu]][[hc]]
   {
      Joiner{}(x, y);
   }

   //! alias for operator T()
   RAJA_INLINE
   T get() { return operator T(); }

//   RAJA_HOST_DEVICE
//   void combine(T other) const { Joiner{}(data->value, other); }

// group reduction out of LDS memory
   RAJA_INLINE void reduce(T x) const [[hc]]
   {
      const auto local = hc_get_workitem_id(0);
      const auto tile = hc_get_group_id(0);

      tile_static T buffer[tile_size];  // use LDS memory to store values for 
                                        // the group

      if (local == 0) std::fill(buffer, buffer+tile_size, Joiner::identity());
      barrier();

      join(buffer[local], x);
      barrier();

      // Reduce within a tile using multiple threads.
      for(std::size_t s = tile_size/2; s > 0; s /= 2)
      {
         if (local < s)
         {
            join(buffer[local], buffer[local+s]);
         }
         barrier();
      }

      // Store the tile result in the global memory.
      if (local == 0)
      {
         typedef  struct RAJA::rocm::detail::rocmInfo RI;
         RI ** ptr= (RI **)((unsigned long)hc::get_dynamic_group_segment_base_pointer()+8);
         join(((T *)(*ptr)->device_mem_ptr)[tile], buffer[0]);
      }
      barrier();
   }
};
#endif

}  // end namespace rocm

//! specialization of ReduceSum for rocm_reduce
template <size_t BLOCK_SIZE, bool Async, bool maybe_atomic, typename T>
class ReduceSum<rocm_reduce<BLOCK_SIZE, Async, maybe_atomic>, T>
    :  public rocm::reducer<RAJA::reduce::sum<T>,T>
{

public:
  //! enable operator+= for ReduceSum 
//      RAJA_INLINE 
//      RAJA_DEVICE
//      explicit ReduceSum(T init_val)
//      : base_type(init_val)
//      {}
//  using Base = rocm::reducer<Async, RAJA::reduce::sum<T>, T, maybe_atomic>;
  using Base = rocm::reducer< RAJA::reduce::sum<T>, T>;
  using Base::Base;

      RAJA_INLINE 
      RAJA_DEVICE
      const ReduceSum<rocm_reduce<BLOCK_SIZE, Async, maybe_atomic>, T>&
      operator+=(T val) const [[hc]]
      {
         this->reduce(val);
         return *this ;
      }

};
#if 0
//! specialization of ReduceMin for rocm_reduce
template <size_t BLOCK_SIZE, bool Async, bool maybe_atomic, typename T>
class ReduceMin<rocm_reduce<BLOCK_SIZE, Async, maybe_atomic>, T>
    :  public hcc::reducer<T, hcc::min<T>>
//    : public rocm::Reduce<Async, RAJA::reduce::min<T>, T, maybe_atomic>
{

public:
  using Base = rocm::Reduce<Async, RAJA::reduce::min<T>, T, maybe_atomic>;
  using Base::Base;
  //! enable min() for ReduceMin -- alias for combine()
  RAJA_HOST_DEVICE
  const ReduceMin& min(T rhs) const
  {
    this->combine(rhs);
    return *this;
  }
};

//! specialization of ReduceMax for rocm_reduce
template <size_t BLOCK_SIZE, bool Async, bool maybe_atomic, typename T>
class ReduceMax<rocm_reduce<BLOCK_SIZE, Async, maybe_atomic>, T>
    : public rocm::Reduce<Async, RAJA::reduce::max<T>, T, maybe_atomic>
{

public:
  using Base = rocm::Reduce<Async, RAJA::reduce::max<T>, T, maybe_atomic>;
  using Base::Base;
  //! enable max() for ReduceMax -- alias for combine()
  RAJA_HOST_DEVICE
  const ReduceMax& max(T rhs) const
  {
    this->combine(rhs);
    return *this;
  }
};

//! specialization of ReduceMinLoc for rocm_reduce
template <size_t BLOCK_SIZE, bool Async, bool maybe_atomic, typename T>
class ReduceMinLoc<rocm_reduce<BLOCK_SIZE, Async, maybe_atomic>, T>
    : public rocm::Reduce<Async,
                          RAJA::reduce::min<RAJA::reduce::detail::ValueLoc<T>>,
                          RAJA::reduce::detail::ValueLoc<T>,
                          maybe_atomic>
{

public:
  using value_type = RAJA::reduce::detail::ValueLoc<T>;
  using Base = rocm::
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

//! specialization of ReduceMaxLoc for rocm_reduce
template <size_t BLOCK_SIZE, bool Async, bool maybe_atomic, typename T>
class ReduceMaxLoc<rocm_reduce<BLOCK_SIZE, Async, maybe_atomic>, T>
    : public rocm::
          Reduce<Async,
                 RAJA::reduce::max<RAJA::reduce::detail::ValueLoc<T, false>>,
                 RAJA::reduce::detail::ValueLoc<T, false>,
                 maybe_atomic>
{
public:
  using value_type = RAJA::reduce::detail::ValueLoc<T, false>;
  using Base = rocm::
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
#endif

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_ROCM guard

#endif  // closing endif for header file include guard
