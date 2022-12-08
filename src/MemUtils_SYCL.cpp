/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for routines used to manage
 *          memory for SYCL reductions and other operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "RAJA/policy/sycl/MemUtils_SYCL.hpp"


namespace RAJA
{

namespace sycl
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

//! State of the host code globally
syclInfo g_status;

//! State of the host code in this thread
syclInfo tl_status;
#if defined(RAJA_ENABLE_OPENMP)
#pragma omp threadprivate(tl_status)
#endif

camp::resources::Resource* app_q = NULL;

void setQueue(camp::resources::Resource* qu) {
  app_q = qu;
}

//! State of raja sycl queue synchronization for sycl reducer objects
std::unordered_map<cl::sycl::queue, bool> g_queue_info_map{
    {cl::sycl::queue(), true}};

cl::sycl::queue* getQueue() {
  if (app_q != NULL) {
    return app_q->get<camp::resources::Sycl>().get_queue();
  }

  return NULL;
}

}  // namespace detail

}  // namespace sycl

}  // namespace RAJA


#endif  // if defined(RAJA_ENABLE_SYCL)
