/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for CUDA hyperplane executors.
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

#ifndef RAJA_policy_cuda_kernel_Hyperplane_HPP
#define RAJA_policy_cuda_kernel_Hyperplane_HPP

#include "RAJA/config.hpp"
#include "RAJA/pattern/kernel/Hyperplane.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"

#include "camp/camp.hpp"

#include <iostream>
#include <type_traits>

namespace RAJA
{
namespace internal
{


template <typename Data,
          camp::idx_t HpArgumentId,
          camp::idx_t... Args,
          typename ExecPolicy,
          typename... EnclosedStmts,
          typename IndexCalc>
struct CudaStatementExecutor<Data,
                             statement::Hyperplane<HpArgumentId,
                                                   cuda_seq_syncthreads_exec,
                                                   ArgList<Args...>,
                                                   ExecPolicy,
                                                   EnclosedStmts...>,
                             IndexCalc> {

  // Add a Collapse policy around our enclosed statements that will handle
  // the inner hyperplane loop's execution
  using stmt_list_t =
      StatementList<statement::Collapse<ExecPolicy,
                                        ArgList<Args...>,
                                        HyperplaneInner<HpArgumentId,
                                                        ArgList<Args...>,
                                                        EnclosedStmts...> > >;

  using enclosed_stmts_t =
      CudaStatementListExecutor<Data, stmt_list_t, IndexCalc>;
  enclosed_stmts_t enclosed_stmts;

  inline RAJA_DEVICE void exec(Data &data,
                               int num_logical_blocks,
                               int block_carry)
  {
    // compute manhattan distance of iteration space to determine
    // as:  hp_len = l0 + l1 + l2 + ...
    int hp_len = segment_length<HpArgumentId>(data)
                 + VarOps::foldl(RAJA::operators::plus<int>(),
                                 segment_length<Args>(data)...);


    /* Execute the outer loop over hyperplanes
     *
     * This will store h in the index_tuple as argument HpArgumentId, so that
     * later, the HyperplaneInner executor can pull it out, and calculate that
     * arguments actual value (and restrict to valid hyperplane indices)
     */
    for (int h = 0; h < hp_len; ++h) {
      data.template assign_offset<HpArgumentId>(h);

      // execute enclosed statements
      enclosed_stmts.exec(data, num_logical_blocks, block_carry);

      __syncthreads();
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


template <typename Data,
          camp::idx_t HpArgumentId,
          camp::idx_t... Args,
          typename... EnclosedStmts,
          typename IndexCalc>
struct CudaStatementExecutor<Data,
                             HyperplaneInner<HpArgumentId,
                                             ArgList<Args...>,
                                             EnclosedStmts...>,
                             IndexCalc> {

  // Add a Collapse policy around our enclosed statements that will handle
  // the inner hyperplane loop's execution
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
    // get h value
    auto h = camp::get<HpArgumentId>(data.offset_tuple);
    using idx_t = decltype(h);

    // get length of Hp indexed argument
    auto len = segment_length<HpArgumentId>(data);


    if (block_carry <= 0) {
      // set indices to beginning of each segment, and increment
      // to this threads first iteration
      bool done = index_calc.assignBegin(data, threadIdx.x, blockDim.x);

      while (!done) {

        // compute actual iterate for HpArgumentId
        // as:  i0 = h - (i1 + i2 + i3 + ...)
        idx_t i = h - VarOps::foldl(RAJA::operators::plus<idx_t>(),
                                    camp::get<Args>(data.offset_tuple)...);

        // check bounds
        if (i >= 0 && i < len) {

          // store in tuple
          data.template assign_offset<HpArgumentId>(i);

          // execute enclosed statements
          enclosed_stmts.exec(data, num_logical_blocks, block_carry);

          // reset h for next iteration
          data.template assign_offset<HpArgumentId>(h);
        }


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


}  // end namespace internal

}  // end namespace RAJA

#endif /* RAJA_pattern_kernel_HPP */
