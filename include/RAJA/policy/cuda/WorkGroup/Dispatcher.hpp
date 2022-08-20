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

namespace cuda
{

// global function that creates the invoker on the device using the
// factory and writes it into a pinned ptr
template < typename invoker_type, typename InvokerFactory >
__global__ void get_invoker_global(
    invoker_type* ptr, InvokerFactory invokerFactory)
{
  *ptr = invokerFactory();
}

// get the pinned ptr buffer
inline void* get_cached_invoker_ptr(size_t nbytes)
{
  static size_t cached_nbytes = 0;
  static void* ptr = nullptr;
  if (nbytes > cached_nbytes) {
    cached_nbytes = 0;
    cudaErrchk(cudaFreeHost(ptr));
    cudaErrchk(cudaMallocHost(&ptr, nbytes));
    cached_nbytes = nbytes;
  }
  return ptr;
}

// mutex that guards against concurrent use of
// pinned buffer and get_cached_invoker_ptr()
inline std::mutex& get_invoker_mutex()
{
  static std::mutex s_mutex;
  return s_mutex;
}

// get the device function pointer by calling a global function to
// write it into a pinned ptr, beware different instantiates of this
// function may run concurrently
template < typename invoker_type, typename InvokerFactory >
inline auto get_invoker(InvokerFactory&& invokerFactory)
{
  const std::lock_guard<std::mutex> lock(get_invoker_mutex());

  auto ptr = static_cast<invoker_type*>(get_cached_invoker_ptr(sizeof(invoker_type)));
  get_invoker_global<invoker_type, std::decay_t<InvokerFactory>><<<1,1>>>(
      ptr, std::forward<InvokerFactory>(invokerFactory));
  cudaErrchk(cudaGetLastError());
  cudaErrchk(cudaDeviceSynchronize());

  return *ptr;
}

// get the device function pointer and store it so it can be used
// multiple times
template < typename invoker_type, typename InvokerFactory >
inline auto get_cached_invoker(InvokerFactory&& invokerFactory)
{
  static auto invoker = get_invoker<invoker_type>(std::forward<InvokerFactory>(invokerFactory));
  return invoker;
}

}  // namespace cuda

/*!
* Populate and return a Dispatcher object that can be used in device code
*/
template < typename T, typename Dispatcher_T, size_t BLOCK_SIZE, size_t BLOCKS_PER_SM, bool Async >
inline const Dispatcher_T* get_Dispatcher(cuda_work_explicit<BLOCK_SIZE, BLOCKS_PER_SM, Async> const&)
{
  using invoker_type = typename Dispatcher_T::invoker_type;
  static Dispatcher_T dispatcher{
        Dispatcher_T::template makeDeviceDispatcher<T>(
          [](auto&& invokerFactory) {
            return cuda::get_cached_invoker<invoker_type>(
              std::forward<decltype(invokerFactory)>(invokerFactory));
          }) };
  return &dispatcher;
}

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
