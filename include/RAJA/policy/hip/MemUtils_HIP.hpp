/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file defining prototypes for routines used to manage
 *          memory for HIP reductions and other operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_MemUtils_HIP_HPP
#define RAJA_MemUtils_HIP_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <type_traits>
#include <unordered_map>

#include "RAJA/util/basic_mempool.hpp"
#include "RAJA/util/mutex.hpp"
#include "RAJA/util/types.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/resource.hpp"

#include "RAJA/policy/hip/policy.hpp"
#include "RAJA/policy/hip/raja_hiperrchk.hpp"

#if defined(RAJA_ENABLE_ROCTX)
#include "hip/hip_runtime_api.h"
#include "roctx.h"
#endif

namespace RAJA
{

namespace hip
{


//! Allocator for pinned memory for use in basic_mempool
struct PinnedAllocator {

  // returns a valid pointer on success, nullptr on failure
  void* malloc(size_t nbytes)
  {
    void* ptr;
    hipErrchk(hipHostMalloc(&ptr, nbytes, hipHostMallocMapped));
    return ptr;
  }

  // returns true on success, false on failure
  bool free(void* ptr)
  {
    hipErrchk(hipHostFree(ptr));
    return true;
  }
};

//! Allocator for device memory for use in basic_mempool
struct DeviceAllocator {

  // returns a valid pointer on success, nullptr on failure
  void* malloc(size_t nbytes)
  {
    void* ptr;
    hipErrchk(hipMalloc(&ptr, nbytes));
    return ptr;
  }

  // returns true on success, false on failure
  bool free(void* ptr)
  {
    hipErrchk(hipFree(ptr));
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
    hipErrchk(hipMalloc(&ptr, nbytes));
    hipErrchk(hipMemset(ptr, 0, nbytes));
    return ptr;
  }

  // returns true on success, false on failure
  bool free(void* ptr)
  {
    hipErrchk(hipFree(ptr));
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
struct hipInfo {
  hip_dim_t gridDim = 0;
  hip_dim_t blockDim = 0;
  ::RAJA::resources::Hip res{::RAJA::resources::Hip::HipFromStream(0,0)};
  bool setup_reducers = false;
#if defined(RAJA_ENABLE_OPENMP)
  hipInfo* thread_states = nullptr;
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

extern hipInfo g_status;

extern hipInfo tl_status;
#if defined(RAJA_ENABLE_OPENMP)
#pragma omp threadprivate(tl_status)
#endif

// stream to synchronization status: true synchronized, false running
extern std::unordered_map<hipStream_t, bool> g_stream_info_map;

RAJA_INLINE
void synchronize_impl(::RAJA::resources::Hip res)
{
  res.wait();
}

}  // namespace detail

//! Ensure all resources in use are synchronized wrt raja kernel launches
RAJA_INLINE
void synchronize()
{
#if defined(RAJA_ENABLE_OPENMP)
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
    hipErrchk(hipDeviceSynchronize());
  }
}

//! Ensure resource is synchronized wrt raja kernel launches
RAJA_INLINE
void synchronize(::RAJA::resources::Hip res)
{
#if defined(RAJA_ENABLE_OPENMP)
  lock_guard<omp::mutex> lock(detail::g_status.lock);
#endif
  auto iter = detail::g_stream_info_map.find(res.get_stream());
  if (iter != detail::g_stream_info_map.end()) {
    if (!iter->second) {
      iter->second = true;
      detail::synchronize_impl(res);
    }
  } else {
    RAJA_ABORT_OR_THROW("Cannot synchronize unknown resource.");
  }
}

//! Indicate resource synchronization status
RAJA_INLINE
void launch(::RAJA::resources::Hip res, bool async = true)
{
#if defined(RAJA_ENABLE_OPENMP)
  lock_guard<omp::mutex> lock(detail::g_status.lock);
#endif
  auto iter = detail::g_stream_info_map.find(res.get_stream());
  if (iter != detail::g_stream_info_map.end()) {
    iter->second = !async;
  } else {
    detail::g_stream_info_map.emplace(res.get_stream(), !async);
  }
  if (!async) {
    detail::synchronize_impl(res);
  }
}

//! Launch kernel and indicate resource synchronization status
RAJA_INLINE
void launch(const void* func, hip_dim_t gridDim, hip_dim_t blockDim, void** args, size_t shmem,
            ::RAJA::resources::Hip res, bool async = true, const char *name = nullptr)
{
  #if defined(RAJA_ENABLE_ROCTX)
  if(name) roctxRangePush(name);
  #else
    RAJA_UNUSED_VAR(name);
  #endif
  hipErrchk(hipLaunchKernel(func, dim3(gridDim), dim3(blockDim), args, shmem, res.get_stream()));
  #if defined(RAJA_ENABLE_ROCTX)
  if(name) roctxRangePop();
  #endif
  launch(res, async);
}

//! Check for errors
RAJA_INLINE
void peekAtLastError() { hipErrchk(hipPeekAtLastError()); }

//! query whether reducers in this thread should setup for device execution now
RAJA_INLINE
bool setupReducers() { return detail::tl_status.setup_reducers; }

//! get gridDim of current launch
RAJA_INLINE
hip_dim_t currentGridDim() { return detail::tl_status.gridDim; }

//! get blockDim of current launch
RAJA_INLINE
hip_dim_t currentBlockDim() { return detail::tl_status.blockDim; }

//! get resource for current launch
RAJA_INLINE
::RAJA::resources::Hip currentResource() { return detail::tl_status.res; }

//! create copy of loop_body that is setup for device execution
template <typename LOOP_BODY>
RAJA_INLINE typename std::remove_reference<LOOP_BODY>::type make_launch_body(
    hip_dim_t gridDim,
    hip_dim_t blockDim,
    size_t RAJA_UNUSED_ARG(dynamic_smem),
    ::RAJA::resources::Hip res,
    LOOP_BODY&& loop_body)
{
  detail::SetterResetter<bool> setup_reducers_srer(
      detail::tl_status.setup_reducers, true);
  detail::SetterResetter<::RAJA::resources::Hip> res_srer(
      detail::tl_status.res, res);

  detail::tl_status.gridDim = gridDim;
  detail::tl_status.blockDim = blockDim;

  using return_type = typename std::remove_reference<LOOP_BODY>::type;
  return return_type(std::forward<LOOP_BODY>(loop_body));
}

RAJA_INLINE
hipDeviceProp_t get_device_prop()
{
  int device;
  hipErrchk(hipGetDevice(&device));
  hipDeviceProp_t prop;
  hipErrchk(hipGetDeviceProperties(&prop, device));
  return prop;
}

RAJA_INLINE
hipDeviceProp_t& device_prop()
{
  static hipDeviceProp_t prop = get_device_prop();
  return prop;
}

}  // namespace hip

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_HIP

#endif  // closing endif for header file include guard
