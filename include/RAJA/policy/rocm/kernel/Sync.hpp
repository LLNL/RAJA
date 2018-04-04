/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run forallN
 *          traversals on GPU with ROCM.
 *
 ******************************************************************************
 */

#ifndef RAJA_policy_rocm_kernel_Sync_HPP
#define RAJA_policy_rocm_kernel_Sync_HPP

#include "RAJA/config.hpp"
#include "RAJA/pattern/kernel.hpp"
#include "camp/camp.hpp"

#if defined(RAJA_ENABLE_ROCM)

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#include <cassert>
#include <climits>

#include "RAJA/config.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"


namespace RAJA
{
namespace statement
{

/*!
 * A kernel::forall statement that performs a ROCM __syncthreads().
 *
 *
 */
struct ROCmSyncThreads : public internal::Statement<camp::nil> {
};

}  // namespace statement

namespace internal
{

template <typename Data, typename IndexCalc>
struct ROCmStatementExecutor<Data, statement::ROCmSyncThreads, IndexCalc> {

  inline __device__ void exec(Data &, int, int) { __syncthreads(); }

  inline RAJA_DEVICE void initBlocks(Data &data,
                                     int num_logical_blocks,
                                     int block_stride)
  {
    // nop
  }


  RAJA_INLINE
  LaunchDim calculateDimensions(Data const &data, LaunchDim const &max_physical)
  {

    return LaunchDim();
  }
};


}  // namespace internal
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_ROCM guard

#endif  // closing endif for header file include guard
