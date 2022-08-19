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

#ifndef RAJA_cuda_WorkGroup_Dispatcher_HPP
#define RAJA_cuda_WorkGroup_Dispatcher_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/cuda/policy.hpp"

#include "RAJA/pattern/WorkGroup/Dispatcher.hpp"

#include <thread>
#include <mutex>


namespace RAJA
{

namespace detail
{

// global function that gets the device function pointer and
// writes it into a pinned ptrptr
template < typename T, typename Dispatcher_T >
__global__ void get_Dispatcher_cuda_device_call_global(
    typename Dispatcher_T::call_sig* ptrptr)
{
  *ptrptr = &Dispatcher_T::template device_call<T>;
}

// allocate the pinned ptrptr buffer
inline void* get_Dispatcher_cuda_device_call_ptrptr()
{
  void* ptrptr = nullptr;
  cudaErrchk(cudaMallocHost(&ptrptr, sizeof(typename Dispatcher<void>::call_sig)));
  return ptrptr;
}

// get the pinned ptrptr buffer
inline void* get_cached_Dispatcher_cuda_device_call_ptrptr()
{
  static void* ptrptr = get_Dispatcher_cuda_device_call_ptrptr();
  return ptrptr;
}

// mutex that guards against concurrent use of
// get_cached_Dispatcher_cuda_device_call_ptrptr()
inline std::mutex& get_Dispatcher_cuda_mutex()
{
  static std::mutex s_mutex;
  return s_mutex;
}

// get the device function pointer by calling a global function to
// write it into a pinned ptrptr, beware different instantiates of this
// function may run concurrently
template < typename T, typename Dispatcher_T >
inline typename Dispatcher_T::call_sig get_Dispatcher_cuda_device_call()
{
  const std::lock_guard<std::mutex> lock(get_Dispatcher_cuda_mutex());

  typename Dispatcher_T::call_sig* ptrptr =
      static_cast<typename Dispatcher_T::call_sig*>(
        get_cached_Dispatcher_cuda_device_call_ptrptr());
  get_Dispatcher_cuda_device_call_global<T, Dispatcher_T><<<1,1>>>(ptrptr);
  cudaErrchk(cudaGetLastError());
  cudaErrchk(cudaDeviceSynchronize());

  return *ptrptr;
}

// get the device function pointer and store it so it can be used
// multiple times
template < typename T, typename Dispatcher_T >
inline typename Dispatcher_T::call_sig get_cached_Dispatcher_cuda_device_call()
{
  static typename Dispatcher_T::call_sig ptr =
      get_Dispatcher_cuda_device_call<T, Dispatcher_T>();
  return ptr;
}

/*!
* Populate and return a Dispatcher object where the
* call operator is a device function
*/
template < typename T, typename Dispatcher_T, size_t BLOCK_SIZE, size_t BLOCKS_PER_SM, bool Async >
inline const Dispatcher_T* get_Dispatcher(cuda_work_explicit<BLOCK_SIZE, BLOCKS_PER_SM, Async> const&)
{
  static Dispatcher_T dispatcher{
        &Dispatcher_T::template move_construct_destroy<T>,
        get_cached_Dispatcher_cuda_device_call<T, Dispatcher_T>(),
        &Dispatcher_T::template destroy<T>,
        sizeof(T)
      };
  return &dispatcher;
}

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
