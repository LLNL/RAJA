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
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
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
#if defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)
#pragma omp threadprivate(tl_status)
#endif

//! State of raja cuda stream synchronization for cuda reducer objects
std::unordered_map<cl::sycl::queue, bool> g_stream_info_map{
    {cl::sycl::queue(), true}};

cl::sycl::queue getQueue() {
  return g_status.stream;
}

}  // namespace detail

}  // namespace sycl

}  // namespace RAJA


#endif  // if defined(RAJA_ENABLE_SYCL)
