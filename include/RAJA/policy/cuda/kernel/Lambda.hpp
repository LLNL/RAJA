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

#ifndef RAJA_policy_cuda_kernel_Lambda_HPP
#define RAJA_policy_cuda_kernel_Lambda_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include <cassert>
#include <climits>

#include "camp/camp.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/kernel.hpp"
#include "RAJA/pattern/kernel/Lambda.hpp"


namespace RAJA
{
namespace internal
{

template <typename Data, camp::idx_t LoopIndex, typename IndexCalc>
struct CudaStatementExecutor<Data, statement::Lambda<LoopIndex>, IndexCalc> {

  IndexCalc index_calc;

  inline __device__ void exec(Data &data,
                              int num_logical_blocks,
                              int block_carry)
  {

    if (block_carry <= 0) {
      // set indices to beginning of each segment, and increment
      // to this threads first iteration
      bool done = index_calc.reset(data);

      while (!done) {

        invoke_lambda<LoopIndex>(data);

        done = index_calc.increment(data);
      }
    }
  }


  inline RAJA_HOST_DEVICE void initBlocks(Data &data,
                                     int num_logical_blocks,
                                     int block_stride)
  {
    // nop
  }

  inline RAJA_DEVICE void initThread(Data &data)
  {
    index_calc.initThread(data, threadIdx.x, blockDim.x);
  }


  RAJA_INLINE
  LaunchDim calculateDimensions(Data const &data, LaunchDim const &max_physical)
  {

    return LaunchDim();
  }
};


template <typename Data, camp::idx_t LoopIndex, typename Segments>
struct CudaStatementExecutor<Data,
                             statement::Lambda<LoopIndex>,
                             CudaIndexCalc_Terminator<Segments>> {

  inline __device__ void exec(Data &data,
                              int num_logical_blocks,
                              int block_carry)

  {
    if (block_carry <= 0) {
      invoke_lambda<LoopIndex>(data);
    }
  }

  inline RAJA_HOST_DEVICE void initBlocks(Data &data,
                                     int num_logical_blocks,
                                     int block_stride)
  {
    // nop
  }

  inline RAJA_DEVICE void initThread(Data &data)
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

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
