/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run kernel
 *          traversals on GPU with CUDA.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_cuda_kernel_Sync_HPP
#define RAJA_policy_cuda_kernel_Sync_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include <cassert>
#include <climits>

#include "camp/camp.hpp"

#include "RAJA/pattern/kernel.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"


namespace RAJA
{
namespace statement
{

/*!
 * A RAJA::kernel statement that performs a CUDA __syncthreads().
 */
struct CudaSyncThreads : public internal::Statement<camp::nil> {
};

/*!
 * A RAJA::kernel statement that performs a CUDA __syncwarp().
 */
struct CudaSyncWarp : public internal::Statement<camp::nil> {
};


}  // namespace statement

namespace internal
{

template <typename Data, typename Types>
struct CudaStatementExecutor<Data, statement::CudaSyncThreads, Types> {

  static
  inline
  RAJA_DEVICE
  void exec(Data &, bool) { __syncthreads(); }


  static
  inline
  LaunchDims calculateDimensions(Data const & RAJA_UNUSED_ARG(data))
  {
    return LaunchDims();
  }
};

template <typename Data, typename Types>
struct CudaStatementExecutor<Data, statement::CudaSyncWarp, Types> {

  static
  inline
  RAJA_DEVICE
#if CUDART_VERSION >= 9000
  void exec(Data &, bool) { __syncwarp(); }
#else
  void exec(Data &, bool) {  }
#endif


  static
  inline
  LaunchDims calculateDimensions(Data const & RAJA_UNUSED_ARG(data))
  {
    return LaunchDims();
  }
};


}  // namespace internal
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
