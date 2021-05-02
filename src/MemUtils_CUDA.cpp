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

#if defined(RAJA_ENABLE_CUDA)

#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"

#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"


#if defined(RAJA_ENABLE_OPENMP) && !defined(_OPENMP)
#error RAJA configured with ENABLE_OPENMP, but OpenMP not supported by current compiler
#endif


namespace RAJA
{

namespace cuda
{

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
#if defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)
omp::mutex g_lock;
#endif

//! Launch info for this thread
LaunchInfo* tl_launch_info = nullptr;
#if defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)
#pragma omp threadprivate(tl_launch_info)
#endif

//! State of raja cuda stream synchronization for cuda reducer objects
std::unordered_map<cudaStream_t, bool> g_stream_info_map{
    {cudaStream_t(0), true}};

}  // namespace detail


void synchronize()
{
#if defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)
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
    cudaErrchk(cudaDeviceSynchronize());
  }
}

void synchronize(cudaStream_t stream)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)
  lock_guard<omp::mutex> lock(detail::g_lock);
#endif
  auto iter = detail::g_stream_info_map.find(stream);
  if (iter != detail::g_stream_info_map.end()) {
    if (!iter->second) {
      iter->second = true;
      cudaErrchk(cudaStreamSynchronize(stream));
    }
  } else {
    fprintf(stderr, "Cannot synchronize unknown stream.\n");
    std::abort();
  }
}

void launch(cudaStream_t stream)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)
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

}  // namespace cuda

}  // namespace RAJA


#endif  // if defined(RAJA_ENABLE_CUDA)
