/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run kernel
 *          traversals on GPU with HIP.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#ifndef RAJA_policy_hip_kernel_Sync_HPP
#define RAJA_policy_hip_kernel_Sync_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include <cassert>
#include <climits>

#include "RAJA/pattern/kernel.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"
#include "camp/camp.hpp"


namespace RAJA
{
namespace statement
{

/*!
 * A RAJA::kernel statement that performs a HIP __syncthreads().
 */
struct HipSyncThreads : public internal::Statement<camp::nil> {
};

/*!
 * A RAJA::kernel statement that performs a HIP __syncwarp().
 */
struct HipSyncWarp : public internal::Statement<camp::nil> {
};

}  // namespace statement

namespace internal
{

template <typename Data, typename Types>
struct HipStatementExecutor<Data, statement::HipSyncThreads, Types> {

  static inline RAJA_DEVICE void exec(Data &, bool) { __syncthreads(); }


  static inline LaunchDims calculateDimensions(
      Data const &RAJA_UNUSED_ARG(data))
  {
    return LaunchDims();
  }
};

template <typename Data, typename Types>
struct HipStatementExecutor<Data, statement::HipSyncWarp, Types> {

  static inline RAJA_DEVICE
      // not currently supported
      void
      exec(Data &, bool)
  {
  }

  static inline LaunchDims calculateDimensions(
      Data const &RAJA_UNUSED_ARG(data))
  {
    return LaunchDims();
  }
};


}  // namespace internal
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_HIP guard

#endif  // closing endif for header file include guard
