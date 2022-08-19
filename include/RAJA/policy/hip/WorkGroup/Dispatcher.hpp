/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA workgroup Dispatcher.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_hip_WorkGroup_Dispatcher_HPP
#define RAJA_hip_WorkGroup_Dispatcher_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/hip/policy.hpp"

#include "RAJA/pattern/WorkGroup/Dispatcher.hpp"

#include <thread>
#include <mutex>


namespace RAJA
{

namespace detail
{

#if defined(RAJA_ENABLE_HIP_INDIRECT_FUNCTION_CALL)

// global function that gets the device function pointer and
// writes it into a pinned ptrptr
template < typename T, typename Dispatcher_T >
__global__ void get_Dispatcher_hip_device_call_global(
    typename Dispatcher_T::call_sig* ptrptr)
{
  *ptrptr = &Dispatcher_T::template device_call<T>;
}

// allocate the pinned ptrptr buffer
inline void* get_Dispatcher_hip_device_call_ptrptr()
{
  void* ptrptr = nullptr;
  hipErrchk(hipHostMalloc(&ptrptr, sizeof(typename Dispatcher<void>::call_sig)));
  return ptrptr;
}

// get the pinned ptrptr buffer
inline void* get_cached_Dispatcher_hip_device_call_ptrptr()
{
  static void* ptrptr = get_Dispatcher_hip_device_call_ptrptr();
  return ptrptr;
}

// mutex that guards against concurrent use of
// get_cached_Dispatcher_hip_device_call_ptrptr()
inline std::mutex& get_Dispatcher_hip_mutex()
{
  static std::mutex s_mutex;
  return s_mutex;
}

template < typename T, typename Dispatcher_T >
inline typename Dispatcher_T::call_sig get_Dispatcher_hip_device_call()
{
  const std::lock_guard<std::mutex> lock(get_Dispatcher_hip_mutex());

  typename Dispatcher_T::call_sig* ptrptr =
      static_cast<typename Dispatcher_T::call_sig*>(
        get_cached_Dispatcher_hip_device_call_ptrptr());
  auto func = get_Dispatcher_hip_device_call_global<T, Dispatcher_T>;
  hipLaunchKernelGGL(func,
      dim3(1), dim3(1), 0, 0, ptrptr);
  hipErrchk(hipGetLastError());
  hipErrchk(hipDeviceSynchronize());

  return *ptrptr;
}

template < typename T, typename Dispatcher_T >
inline typename Dispatcher_T::call_sig get_cached_Dispatcher_hip_device_call()
{
  static typename Dispatcher_T::call_sig ptr =
      get_Dispatcher_hip_device_call<T, Dispatcher_T>();
  return ptr;
}

/*!
* Populate and return a Dispatcher object where the
* call operator is a device function
*/
template < typename T, typename Dispatcher_T, size_t BLOCK_SIZE, bool Async >
inline const Dispatcher_T* get_Dispatcher(hip_work<BLOCK_SIZE, Async> const&)
{
  static Dispatcher_T dispatcher{
        &Dispatcher_T::template move_construct_destroy<T>,
        get_cached_Dispatcher_hip_device_call<T, Dispatcher_T>(),
        &Dispatcher_T::template destroy<T>,
        sizeof(T)
      };
  return &dispatcher;
}

#endif

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
