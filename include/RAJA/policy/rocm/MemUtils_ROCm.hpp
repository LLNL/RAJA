/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file defining prototypes for routines used to manage
 *          memory for ROCM reductions and other operations.
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

#ifndef RAJA_MemUtils_ROCM_HPP
#define RAJA_MemUtils_ROCM_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_ROCM)

#include "RAJA/util/types.hpp"

#include "RAJA/util/basic_mempool.hpp"

#include "RAJA/policy/rocm/atomic.hpp"

#include "RAJA/policy/rocm/raja_rocmerrchk.hpp"

#include "RAJA/util/mutex.hpp"

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <type_traits>
#include <unordered_map>

/*
**  These moved outside of RAJA namespace because the cuda equivalents
**  are also outside.
*/
rocmError_t rocmHostAlloc(void ** ptr , size_t nbytes, int device = 0);
rocmError_t rocmFreeHost(void * ptr);

void * rocmDeviceAlloc(size_t nbytes, int device = 0);
rocmError_t rocmMalloc(void ** ptr, size_t nbytes, int device = 0);
rocmError_t rocmMallocManaged(void ** ptr, size_t nbytes, int device = 0);
rocmError_t rocmDeviceFree(void * ptr);
rocmError_t rocmFree(void * ptr);
rocmError_t rocmMemset(void * ptr, unsigned char value, size_t nbytes);
rocmError_t rocmMemcpy(void * src, void * dst, size_t size);


namespace RAJA
{

RAJA_INLINE
rocmError_t rocmDeviceSynchronize()
{
  hc::accelerator_view av = hc::accelerator().get_default_view();
  hc::completion_future fut = av.create_marker();
  fut.wait();
  return rocmPeekAtLastError();
}

#if __KALMAR_ACCELERATOR__ == 1
RAJA_DEVICE RAJA_INLINE
void __syncthreads() [[hc]]
{
   amp_barrier(CLK_LOCAL_MEM_FENCE);
}

RAJA_DEVICE RAJA_INLINE
void __threadfence() [[hc]]
{
   amp_barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

#define LT0 ((threadIdx_x+threadIdx_y+threadIdx_z)?0:1)


// returns non-zero if and only if predicate is non-zero for all threads
RAJA_DEVICE RAJA_INLINE
int __syncthreads_or(int predicate) [[hc]]
{
  int *shared_var = (int *)hc::get_dynamic_group_segment_base_pointer();;
  if(LT0) *shared_var = 0;
  hc_barrier(CLK_LOCAL_MEM_FENCE);
  if (predicate) RAJA::atomic::atomicOr(RAJA::atomic::rocm_atomic{},shared_var,1);
  hc_barrier(CLK_LOCAL_MEM_FENCE);
  return (*shared_var);
}
#endif

////////////////////////////////////////////////////////////////////
namespace rocm
{
//! Allocator for pinned memory for use in basic_mempool
struct PinnedAllocator {

  // returns a valid pointer on success, nullptr on failure
  void* malloc(size_t nbytes)
  {
    void* ptr;
//see /opt/rocm/hcc/include/hc_am.hpp
    rocmErrchk(rocmHostAlloc(&ptr,nbytes,2));
    return ptr;
  }

  // returns true on success, false on failure
  bool free(void* ptr)
  {
    rocmErrchk(rocmFreeHost(ptr));
    return true;
  }
};

//! Allocator for device memory for use in basic_mempool
struct DeviceAllocator {

  // returns a valid pointer on success, nullptr on failure
  void* malloc(size_t nbytes)
  {
    void* ptr;
    hc::accelerator acc;
    ptr = hc::am_alloc(nbytes,acc,0);
    rocmErrchk(rocmPeekAtLastError());
    return ptr;
  }

  // returns true on success, false on failure
  bool free(void* ptr)
  {
    hc::am_free(ptr);
    rocmErrchk(rocmPeekAtLastError());
    return true;
  }
};


//! Allocator for pre-zeroed device memory for use in basic_mempool
//  Note: Memory must be zero when returned to mempool
struct DeviceZeroedAllocator {

  // returns a valid pointer on success, nullptr on failure
  void* malloc(size_t nbytes)
  {
    void* ptr;
    hc::accelerator acc;
    ptr = hc::am_alloc(nbytes,acc,0);
    rocmErrchk(rocmPeekAtLastError());
    rocmMemset(ptr, 0, nbytes);
    rocmErrchk(rocmPeekAtLastError());
    return ptr;
  }

  // returns true on success, false on failure
  bool free(void* ptr)
  {
    if(!hc::am_free(ptr)) return false;
    return true;
  }
};

using device_mempool_type = basic_mempool::MemPool<DeviceAllocator>;
using device_zeroed_mempool_type =
    basic_mempool::MemPool<DeviceZeroedAllocator>;
using pinned_mempool_type = basic_mempool::MemPool<PinnedAllocator>;

namespace detail
{

//! struct containing data necessary to coordinate kernel launches with reducers
struct rocmInfo {
  dim3 gridDim = 0;
  dim3 blockDim = 0;
  rocmStream_t stream = 0;
  bool setup_reducers = false;
#if defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)
  rocmInfo* thread_states = nullptr;
  omp::mutex lock;
#endif
};

//! class that changes a value on construction then resets it at destruction
template <typename T>
class SetterResetter
{
public:
  SetterResetter(T& val, T new_val) : m_val(val), m_old_val(val)
  {
    m_val = new_val;
  }
  SetterResetter(const SetterResetter&) = delete;
  ~SetterResetter() { m_val = m_old_val; }

private:
  T& m_val;
  T m_old_val;
};

extern rocmInfo g_status;

extern rocmInfo tl_status;
#if defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)
#pragma omp threadprivate(tl_status)
#endif

extern std::unordered_map<rocmStream_t, bool> g_stream_info_map;

}  // closing brace for detail namespace

//! Ensure all streams in use are synchronized wrt raja kernel launches
RAJA_INLINE
void synchronize()
{
#if defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)
  lock_guard<omp::mutex> lock(detail::g_status.lock);
#endif
  bool synchronize = false;
  for (auto& val : detail::g_stream_info_map) {
    if (!val.second) {
      synchronize = true;
      val.second = true;
    }
  }
  if (synchronize) {
    rocmDeviceSynchronize();
  }
}

//! Ensure stream is synchronized wrt raja kernel launches
RAJA_INLINE
void synchronize(hipStream_t stream)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)
  lock_guard<omp::mutex> lock(detail::g_status.lock);
#endif
  auto iter = detail::g_stream_info_map.find(stream);
  if (iter != detail::g_stream_info_map.end()) {
    if (!iter->second) {
      iter->second = true;
      rocmErrchk(hipStreamSynchronize(stream));
    }
  } else {
    fprintf(stderr, "Cannot synchronize unknown stream.\n");
    std::abort();
  }
}

//! Indicate stream is asynchronous
RAJA_INLINE
void launch(hipStream_t stream)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)
  lock_guard<omp::mutex> lock(detail::g_status.lock);
#endif
  auto iter = detail::g_stream_info_map.find(stream);
  if (iter != detail::g_stream_info_map.end()) {
    iter->second = false;
  } else {
    detail::g_stream_info_map.emplace(stream, false);
  }
}


//! query whether reducers in this thread should setup for device execution now
RAJA_INLINE
bool setupReducers() { return detail::tl_status.setup_reducers; }

//! get gridDim of current launch
RAJA_INLINE
dim3 currentGridDim() { return detail::tl_status.gridDim; }

//! get blockDim of current launch
RAJA_INLINE
dim3 currentBlockDim() { return detail::tl_status.blockDim; }

//! get stream for current launch
RAJA_INLINE
hipStream_t currentStream() { return detail::tl_status.stream; }

//! create copy of loop_body that is setup for device execution
template <typename LOOP_BODY>
RAJA_INLINE typename std::remove_reference<LOOP_BODY>::type make_launch_body(
    dim3 gridDim,
    dim3 blockDim,
    size_t dynamic_smem,
    hipStream_t stream,
    LOOP_BODY&& loop_body)
{
  detail::SetterResetter<bool> setup_reducers_srer(
      detail::tl_status.setup_reducers, true);

  detail::tl_status.stream = stream;
  detail::tl_status.gridDim = gridDim;
  detail::tl_status.blockDim = blockDim;

  return {loop_body};
}

}  // closing brace for rocm namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_ROCM

#endif  // closing endif for header file include guard
