/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run forallN
 *          traversals on GPU with CUDA.
 *
 ******************************************************************************
 */

#ifndef RAJA_policy_cuda_kernel_Collapse_HPP
#define RAJA_policy_cuda_kernel_Collapse_HPP

#include "RAJA/config.hpp"
#include "RAJA/pattern/kernel.hpp"
#include "RAJA/policy/cuda/kernel/For.hpp"
#include "camp/camp.hpp"

#if defined(RAJA_ENABLE_CUDA)

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

#include "RAJA/pattern/kernel/Collapse.hpp"
#include "RAJA/policy/cuda/kernel/For.hpp"


namespace RAJA
{
namespace internal
{


/*
 * Statement Executor for collapsing multiple segments iteration space,
 * and provides work sharing according to the collapsing execution policy.
 */
template <typename Data,
          typename ExecPolicy,
          typename... EnclosedStmts,
          typename IndexCalc>
struct CudaStatementExecutor<Data,
                             statement::Collapse<ExecPolicy,
                                                 ArgList<>,
                                                 EnclosedStmts...>,
                             IndexCalc> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  using enclosed_stmts_t =
      CudaStatementListExecutor<Data, stmt_list_t, IndexCalc>;
  enclosed_stmts_t enclosed_stmts;

  inline RAJA_DEVICE void exec(Data &data,
                               int num_logical_blocks,
                               int block_carry)
  {
    // execute enclosed statements
    enclosed_stmts.exec(data, num_logical_blocks, block_carry);
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


/*
 * Statement Executor for collapsing multiple segments iteration space,
 * and provides work sharing according to the collapsing execution policy.
 */
template <typename Data,
          typename ExecPolicy,
          camp::idx_t Arg0,
          camp::idx_t... ArgRest,
          typename... EnclosedStmts,
          typename IndexCalc>
struct CudaStatementExecutor<Data,
                             statement::Collapse<ExecPolicy,
                                                 ArgList<Arg0, ArgRest...>,
                                                 EnclosedStmts...>,
                             IndexCalc> {

  using stmt_list_t =
      StatementList<statement::For<Arg0,
                                   ExecPolicy,
                                   statement::Collapse<ExecPolicy,
                                                       ArgList<ArgRest...>,
                                                       EnclosedStmts...> > >;

  using enclosed_stmts_t =
      CudaStatementListExecutor<Data, stmt_list_t, IndexCalc>;
  enclosed_stmts_t enclosed_stmts;

  inline RAJA_DEVICE void exec(Data &data,
                               int num_logical_blocks,
                               int block_carry)
  {
    // execute enclosed statements
    enclosed_stmts.exec(data, num_logical_blocks, block_carry);
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
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
