/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for routines used to manage
 *          memory for CUDA reductions and other operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include <mutex>

#include "RAJA/policy/hip/MemUtils_HIP.hpp"
#include "RAJA/policy/hip/raja_hiperrchk.hpp"
#include "RAJA/util/AllocatorPool.hpp"


#if defined(RAJA_ENABLE_OPENMP) && !defined(_OPENMP)
#error RAJA configured with ENABLE_OPENMP, but OpenMP not supported by current compiler
#endif


namespace RAJA
{

namespace hip
{

namespace detail
{

//! Allocator for pinned memory for use in AllocatorPool
struct PinnedAllocator
{
  const char* getName() const noexcept
  {
    return "RAJA::hip::detail::PinnedAllocator";
  }
  Platform getPlatform() const noexcept
  {
    return Platform::hip;
  }
  // returns a valid pointer on success, nullptr on failure
  void* allocate(size_t nbytes)
  {
    void* ptr;
    hipErrchk(hipHostMalloc(&ptr, nbytes, hipHostMallocMapped));
    return ptr;
  }

  // returns true on success, false on failure
  void deallocate(void* ptr)
  {
    hipErrchk(hipHostFree(ptr));
  }
};

//! Allocator for device memory for use in AllocatorPool
struct DeviceAllocator
{
  const char* getName() const noexcept
  {
    return "RAJA::hip::detail::DeviceAllocator";
  }
  Platform getPlatform() const noexcept
  {
    return Platform::hip;
  }
  // returns a valid pointer on success, nullptr on failure
  void* allocate(size_t nbytes)
  {
    void* ptr;
    hipErrchk(hipMalloc(&ptr, nbytes));
    return ptr;
  }

  // returns true on success, false on failure
  void deallocate(void* ptr)
  {
    hipErrchk(hipFree(ptr));
  }
};

//! Allocator for pre-zeroed device memory for use in AllocatorPool
//  Note: Memory must be zero when returned from allocate here
//  Note: Memory must be zero when returned to AllocatorPool
struct DeviceZeroedAllocator
{
  const char* getName() const noexcept
  {
    return "RAJA::hip::detail::DeviceZeroedAllocator";
  }
  Platform getPlatform() const noexcept
  {
    return Platform::cuda;
  }
  // returns a valid pointer on success, nullptr on failure
  void* allocate(size_t nbytes)
  {
    void* ptr;
    hipErrchk(hipMalloc(&ptr, nbytes));
    hipErrchk(hipMemset(ptr, 0, nbytes));
    return ptr;
  }

  // returns true on success, false on failure
  void deallocate(void* ptr)
  {
    hipErrchk(hipFree(ptr));
  }
};


static RAJA::Allocator* s_device_allocator = nullptr;
static RAJA::Allocator* s_device_zeroed_allocator = nullptr;
static RAJA::Allocator* s_pinned_allocator = nullptr;

void set_device_allocator(RAJA::Allocator* allocator)
{
  if (s_device_allocator != nullptr) {
    if (s_device_allocator->getAllocationCount() != 0u) {
      RAJA_ABORT_OR_THROW("RAJA::hip::set_device_allocator old pool is not empty");
    }
    s_device_allocator->release();
    RAJA::detail::remove_allocator(s_device_allocator);
    delete s_device_allocator;
  }
  s_device_allocator = allocator;
  if (s_device_allocator != nullptr) {
    RAJA::detail::add_allocator(s_device_allocator);
  }
}

void set_device_zeroed_allocator(RAJA::Allocator* allocator)
{
  if (s_device_zeroed_allocator != nullptr) {
    if (s_device_zeroed_allocator->getAllocationCount() != 0u) {
      RAJA_ABORT_OR_THROW("RAJA::hip::set_device_zeroed_allocator old pool is not empty");
    }
    s_device_zeroed_allocator->release();
    RAJA::detail::remove_allocator(s_device_zeroed_allocator);
    delete s_device_zeroed_allocator;
  }
  s_device_zeroed_allocator = allocator;
  if (s_device_zeroed_allocator != nullptr) {
    RAJA::detail::add_allocator(s_device_zeroed_allocator);
  }
}

void set_pinned_allocator(RAJA::Allocator* allocator)
{
  if (s_pinned_allocator != nullptr) {
    if (s_pinned_allocator->getAllocationCount() != 0u) {
      RAJA_ABORT_OR_THROW("RAJA::hip::set_pinned_allocator old pool is not empty");
    }
    s_pinned_allocator->release();
    RAJA::detail::remove_allocator(s_pinned_allocator);
    delete s_pinned_allocator;
  }
  s_pinned_allocator = allocator;
  if (s_pinned_allocator != nullptr) {
    RAJA::detail::add_allocator(s_pinned_allocator);
  }
}

}  // namespace detail


void reset_device_allocator()
{
  set_device_allocator<
        RAJA::AllocatorPool<detail::DeviceAllocator>
      >();
}

void reset_device_zeroed_allocator()
{
  set_device_zeroed_allocator<
        RAJA::AllocatorPool<detail::DeviceZeroedAllocator>
      >();
}

void reset_pinned_allocator()
{
  set_pinned_allocator<
        RAJA::AllocatorPool<detail::PinnedAllocator>
      >();
}

RAJA::Allocator& get_device_allocator()
{
  static std::once_flag s_onceFlag;
  std::call_once(s_onceFlag, [](){
    if (detail::s_device_allocator == nullptr) {
      reset_device_allocator();
    }
  });
  return *detail::s_device_allocator;
}

RAJA::Allocator& get_device_zeroed_allocator()
{
  static std::once_flag s_onceFlag;
  std::call_once(s_onceFlag, [](){
    if (detail::s_device_zeroed_allocator == nullptr) {
      reset_device_zeroed_allocator();
    }
  });
  return *detail::s_device_zeroed_allocator;
}

RAJA::Allocator& get_pinned_allocator()
{
  static std::once_flag s_onceFlag;
  std::call_once(s_onceFlag, [](){
    if (detail::s_pinned_allocator == nullptr) {
      reset_pinned_allocator();
    }
  });
  return *detail::s_pinned_allocator;
}


namespace detail
{
//
/////////////////////////////////////////////////////////////////////////////
//
// Variables representing the state of execution.
//
/////////////////////////////////////////////////////////////////////////////
//

//! Lock for global state updates
#if defined(RAJA_ENABLE_OPENMP)
omp::mutex g_lock;
#endif

//! Launch info for this thread
LaunchInfo* tl_launch_info = nullptr;
#if defined(RAJA_ENABLE_OPENMP)
#pragma omp threadprivate(tl_launch_info)
#endif

//! State of raja hip stream synchronization for hip reducer objects
std::unordered_map<hipStream_t, bool> g_stream_info_map{
    {hipStream_t(0), true}};

}  // namespace detail


void synchronize()
{
#if defined(RAJA_ENABLE_OPENMP)
  lock_guard<omp::mutex> lock(detail::g_lock);
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

void synchronize(hipStream_t stream)
{
#if defined(RAJA_ENABLE_OPENMP)
  lock_guard<omp::mutex> lock(detail::g_lock);
#endif
  auto iter = detail::g_stream_info_map.find(stream);
  if (iter != detail::g_stream_info_map.end()) {
    if (!iter->second) {
      iter->second = true;
      hipErrchk(hipStreamSynchronize(stream));
    }
  } else {
    fprintf(stderr, "Cannot synchronize unknown stream.\n");
    std::abort();
  }
}

void launch(hipStream_t stream)
{
#if defined(RAJA_ENABLE_OPENMP)
  lock_guard<omp::mutex> lock(detail::g_lock);
#endif
  auto iter = detail::g_stream_info_map.find(stream);
  if (iter != detail::g_stream_info_map.end()) {
    iter->second = false;
  } else {
    detail::g_stream_info_map.emplace(stream, false);
  }
}

detail::LaunchInfo*& get_tl_launch_info()
{
   return detail::tl_launch_info;
}

}  // namespace hip

}  // namespace RAJA


#endif  // if defined(RAJA_ENABLE_HIP)
