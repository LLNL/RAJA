/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for CUDA statement executors.
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


#ifndef RAJA_policy_cuda_kernel_For_HPP
#define RAJA_policy_cuda_kernel_For_HPP

#include "RAJA/config.hpp"
#include "RAJA/policy/cuda/kernel/internal.hpp"


namespace RAJA
{

namespace internal
{


/*
 * Executor for thread work sharing loop inside a Cuda Kernel.
 *
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename... EnclosedStmts,
          typename IndexCalc>
struct CudaStatementExecutor<Data,
                             statement::For<ArgumentId,
                                            cuda_thread_exec,
                                            EnclosedStmts...>,
                             IndexCalc> {

  using stmt_list_t = StatementList<EnclosedStmts...>;
  using index_calc_t =
      ExtendCudaIndexCalc<IndexCalc,
                          CudaIndexCalc_Policy<ArgumentId, cuda_thread_exec>>;

  using enclosed_stmts_t =
      CudaStatementListExecutor<Data, stmt_list_t, index_calc_t>;
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

    LaunchDim dim = enclosed_stmts.calculateDimensions(data, max_physical);

    dim.threads *= segment_length<ArgumentId>(data);

    return dim;
  }
};


/*
 * Executor for block work sharing loop inside a Cuda Kernel.
 *
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename... EnclosedStmts,
          typename IndexCalc>
struct CudaStatementExecutor<Data,
                             statement::For<ArgumentId,
                                            cuda_block_exec,
                                            EnclosedStmts...>,
                             IndexCalc> : public CudaBlockLoop<ArgumentId, 1> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  using enclosed_stmts_t =
      CudaStatementListExecutor<Data, stmt_list_t, IndexCalc>;
  enclosed_stmts_t enclosed_stmts;


  inline RAJA_DEVICE void exec(Data &data,
                               int num_logical_blocks,
                               int block_carry)
  {
    execBlockLoop(*this, data, num_logical_blocks, block_carry);
  }


  inline RAJA_DEVICE void initBlocks(Data &data,
                                     int num_logical_blocks,
                                     int block_stride)
  {
    int len = segment_length<ArgumentId>(data);
    initBlockLoop(enclosed_stmts, data, len, num_logical_blocks, block_stride);
  }


  RAJA_INLINE
  LaunchDim calculateDimensions(Data const &data, LaunchDim const &max_physical)
  {

    LaunchDim dim = enclosed_stmts.calculateDimensions(data, max_physical);

    dim.blocks *= segment_length<ArgumentId>(data);

    return dim;
  }
};


/*
 * Executor for thread and block work sharing loop inside a Cuda Kernel.
 *
 */
template <typename Data,
          camp::idx_t ArgumentId,
          size_t max_threads,
          typename... EnclosedStmts,
          typename IndexCalc>
struct CudaStatementExecutor<Data,
                             statement::For<ArgumentId,
                                            cuda_threadblock_exec<max_threads>,
                                            EnclosedStmts...>,
                             IndexCalc>
    : public CudaBlockLoop<ArgumentId, max_threads> {

  using stmt_list_t = StatementList<EnclosedStmts...>;
  using index_calc_t =
      ExtendCudaIndexCalc<IndexCalc,
                          CudaIndexCalc_Policy<ArgumentId, cuda_thread_exec>>;

  using enclosed_stmts_t =
      CudaStatementListExecutor<Data, stmt_list_t, index_calc_t>;
  enclosed_stmts_t enclosed_stmts;


  inline RAJA_DEVICE void exec(Data &data,
                               int num_logical_blocks,
                               int block_carry)
  {
    execBlockLoop(*this, data, num_logical_blocks, block_carry);
  }


  inline RAJA_DEVICE void initBlocks(Data &data,
                                     int num_logical_blocks,
                                     int block_stride)
  {
    int len = segment_length<ArgumentId>(data);
    initBlockLoop(enclosed_stmts, data, len, num_logical_blocks, block_stride);
  }

  RAJA_INLINE
  LaunchDim calculateDimensions(Data const &data, LaunchDim const &max_physical)
  {

    LaunchDim dim = enclosed_stmts.calculateDimensions(data, max_physical);

    // Compute how many blocks
    int len = segment_length<ArgumentId>(data);
    int num_blocks = len / max_threads;
    if (num_blocks * max_threads < len) {
      num_blocks++;
    }

    dim.blocks *= num_blocks;
    dim.threads *= max_threads;

    return dim;
  }
};


/*
 * Executor for sequential loops inside of a Cuda Kernel.
 *
 * This is specialized since it need to execute the loop immediately.
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename... EnclosedStmts,
          typename IndexCalc>
struct CudaStatementExecutor<Data,
                             statement::
                                 For<ArgumentId, seq_exec, EnclosedStmts...>,
                             IndexCalc> {

  using stmt_list_t = StatementList<EnclosedStmts...>;
  using index_calc_t =
      ExtendCudaIndexCalc<IndexCalc,
                          CudaIndexCalc_Policy<ArgumentId, seq_exec>>;

  using enclosed_stmts_t =
      CudaStatementListExecutor<Data, stmt_list_t, index_calc_t>;
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


template <typename Data,
          camp::idx_t ArgumentId,
          typename... EnclosedStmts,
          typename Segments>
struct CudaStatementExecutor<Data,
                             statement::
                                 For<ArgumentId, seq_exec, EnclosedStmts...>,
                             CudaIndexCalc_Terminator<Segments>> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  using enclosed_stmts_t =
      CudaStatementListExecutor<Data,
                                stmt_list_t,
                                CudaIndexCalc_Terminator<Segments>>;
  enclosed_stmts_t enclosed_stmts;

  inline RAJA_DEVICE void exec(Data &data,
                               int num_logical_blocks,
                               int block_carry)
  {
    int len = segment_length<ArgumentId>(data);

    for (int i = 0; i < len; ++i) {
      data.template assign_offset<ArgumentId>(i);

      // execute enclosed statements
      enclosed_stmts.exec(data, num_logical_blocks, block_carry);
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


/*
 * Executor for sequential loops inside of a Cuda Kernel.
 *
 * This is specialized since it need to execute the loop immediately.
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename... EnclosedStmts,
          typename IndexCalc>
struct CudaStatementExecutor<Data,
                             statement::For<ArgumentId,
                                            cuda_seq_syncthreads_exec,
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
    int len = segment_length<ArgumentId>(data);

    for (int i = 0; i < len; ++i) {
      data.template assign_offset<ArgumentId>(i);

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


}  // namespace internal
}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
