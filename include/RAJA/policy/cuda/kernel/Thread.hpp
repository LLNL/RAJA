/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for CUDA thread executors.
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


#ifndef RAJA_policy_cuda_kernel_Thread_HPP
#define RAJA_policy_cuda_kernel_Thread_HPP


#include "RAJA/config.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/cuda/kernel/internal.hpp"

#include <iostream>
#include <type_traits>

namespace RAJA
{
namespace statement
{

template <typename... EnclosedStmts>
struct Thread : public internal::Statement<EnclosedStmts...> {
};

}  // end namespace statement

namespace internal
{


template <typename Data, typename... EnclosedStmts, typename IndexCalc>
struct CudaStatementExecutor<Data,
                             statement::Thread<EnclosedStmts...>,
                             IndexCalc> {

  using stmt_list_t = StatementList<EnclosedStmts...>;
  using index_calc_t = CudaIndexCalc_Terminator<typename Data::segment_tuple_t>;

  using enclosed_stmts_t =
      CudaStatementListExecutor<Data, stmt_list_t, index_calc_t>;
  enclosed_stmts_t enclosed_stmts;


  IndexCalc index_calc;

  inline RAJA_DEVICE void exec(Data &data,
                               int num_logical_blocks,
                               int block_carry)
  {

    if (block_carry <= 0) {
      // set indices to beginning of each segment, and increment
      // to this threads first iteration
      bool done = index_calc.assignBegin(data, threadIdx.x, blockDim.x);

      while (!done) {

        // execute enclosed statements
        enclosed_stmts.exec(data, num_logical_blocks, block_carry);

        done = index_calc.increment(data, blockDim.x);
      }
    }
  }


  inline RAJA_DEVICE void initBlocks(Data &data,
                                     int num_logical_blocks,
                                     int block_stride)
  {
    enclosed_stmts.initBlocks(data, num_logical_blocks, block_stride);
  }

  RAJA_INLINE
  LaunchDim calculateDimensions(Data const &data, LaunchDim const &max_physical)
  {

    return enclosed_stmts.calculateDimensions(data, max_physical);
  }
};


}  // namespace internal
}  // end namespace RAJA


#endif
