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
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
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
  cl::sycl::queue qu = cl::sycl::queue();
  bool setup_reducers = false;
#if defined(RAJA_ENABLE_OPENMP)
  syclInfo* thread_states = nullptr;
  omp::mutex lock;
#endif
};

extern syclInfo g_status;

extern syclInfo tl_status;

extern std::unordered_map<cl::sycl::queue, bool> g_queue_info_map;

void setQueue(camp::resources::Resource* q);

cl::sycl::queue* getQueue();

}  // namespace detail

}  // namespace sycl

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_SYCL

#endif  // closing endif for header file include guard

