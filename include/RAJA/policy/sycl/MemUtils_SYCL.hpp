/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file defining prototypes for routines used to manage
 *          memory for SYCL reductions and other operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_MemUtils_SYCL_HPP
#define RAJA_MemUtils_SYCL_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <CL/sycl.hpp>

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <type_traits>
#include <unordered_map>

#include "RAJA/util/basic_mempool.hpp"
#include "RAJA/util/mutex.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/sycl/policy.hpp"
//#include "RAJA/policy/sycl/raja_syclerrchk.hpp"

namespace RAJA
{

namespace sycl
{

namespace detail
{

//! struct containing data necessary to coordinate kernel launches with reducers
struct syclInfo {
  sycl_dim_t gridDim{0};
  sycl_dim_t blockDim{0};
  cl::sycl::queue stream = cl::sycl::queue();
  bool setup_reducers = false;
#if defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)
  syclInfo* thread_states = nullptr;
  omp::mutex lock;
#endif
};

extern syclInfo g_status;

extern syclInfo tl_status;

extern std::unordered_map<cl::sycl::queue, bool> g_stream_info_map;

}  // namespace detail


//! Indicate stream is asynchronous
RAJA_INLINE
void launch(cl::sycl::queue stream)
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

//! Launch kernel and indicate stream is asynchronous
RAJA_INLINE
void launch(const void* &func, sycl_dim_t gridDim, sycl_dim_t blockDim, void** args, size_t shmem, cl::sycl::queue stream)
{

  std::cout << "GridDims:\n\tGridDim0: " << gridDim.get(0) << std::endl;
  std::cout << "GridDims:\n\tBlockDim0: " << blockDim.get(0) <<  std::endl;
  auto global_size = blockDim.get(0);
  std::cout << "Ready to Launch" << std::endl;

  cl::sycl::queue q;

  q.submit([&](cl::sycl::handler& h) {
    h.parallel_for( cl::sycl::nd_range<1>{global_size, 1},
                    [=] (cl::sycl::nd_item<1> item) {

      size_t ii = item.get_global_id(0);
//       func;
    });
  });
//  syclErrchk(syclLaunchKernel(func, gridDim, blockDim, args, shmem, stream));
  launch(stream);
}

template <typename LOOP_BODY>
RAJA_INLINE typename std::remove_reference<LOOP_BODY>::type make_launch_body(
    sycl_dim_t gridDim,
    sycl_dim_t blockDim,
    size_t dynamic_smem,
    cl::sycl::queue stream,
    LOOP_BODY&& loop_body)
{
//  detail::SetterResetter<bool> setup_reducers_srer(
  //    detail::tl_status.setup_reducers, true);

  detail::tl_status.stream = stream;
  detail::tl_status.gridDim = gridDim;
  detail::tl_status.blockDim = blockDim;

  using return_type = typename std::remove_reference<LOOP_BODY>::type;
  return return_type(std::forward<LOOP_BODY>(loop_body));
}


}  // namespace sycl

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_SYCL

#endif  // closing endif for header file include guard

