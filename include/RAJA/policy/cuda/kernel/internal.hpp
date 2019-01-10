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
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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


#ifndef RAJA_policy_cuda_kernel_internal_HPP
#define RAJA_policy_cuda_kernel_internal_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include <cassert>
#include <climits>

#include "camp/camp.hpp"

#include "RAJA/pattern/kernel.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#include "RAJA/policy/cuda/policy.hpp"

#include "RAJA/internal/LegacyCompatibility.hpp"


namespace RAJA
{


/*!
 * Policy for For<>, executes loop iteration by distributing them over threads.
 * This does no (additional) work-sharing between thread blocks.
 */

struct cuda_thread_exec : public RAJA::make_policy_pattern_launch_platform_t<
                              RAJA::Policy::cuda,
                              RAJA::Pattern::forall,
                              RAJA::Launch::undefined,
                              RAJA::Platform::cuda> {
};


/*!
 * Policy for For<>, executes loop iteration by distributing iterations
 * exclusively over blocks.
 */

struct cuda_block_exec : public RAJA::make_policy_pattern_launch_platform_t<
                             RAJA::Policy::cuda,
                             RAJA::Pattern::forall,
                             RAJA::Launch::undefined,
                             RAJA::Platform::cuda> {
};


/*!
 * Policy for For<>, executes loop iteration by distributing work over
 * physical blocks and executing sequentially within blocks.
 */

template <size_t num_blocks>
struct cuda_block_seq_exec : public RAJA::make_policy_pattern_launch_platform_t<
                                 RAJA::Policy::cuda,
                                 RAJA::Pattern::forall,
                                 RAJA::Launch::undefined,
                                 RAJA::Platform::cuda> {
};


/*!
 * Policy for For<>, executes loop iteration by distributing them over threads
 * and blocks, but limiting the number of threads to num_threads.
 */
template <size_t num_threads>
struct cuda_threadblock_exec
    : public RAJA::make_policy_pattern_launch_platform_t<
          RAJA::Policy::cuda,
          RAJA::Pattern::forall,
          RAJA::Launch::undefined,
          RAJA::Platform::cuda> {
};


namespace internal
{


struct LaunchDims {

  cuda_dim_t blocks;
  cuda_dim_t threads;
  cuda_dim_t min_threads;

  RAJA_INLINE
  RAJA_HOST_DEVICE
  LaunchDims() : blocks(0,0,0), threads(0,0,0), min_threads(0,0,0) {}


  RAJA_INLINE
  RAJA_HOST_DEVICE
  LaunchDims(LaunchDims const &c) :
  blocks(c.blocks), threads(c.threads), min_threads(c.min_threads)
  {
  }

  RAJA_INLINE
  LaunchDims max(LaunchDims const &c) const
  {
    LaunchDims result;

    result.blocks.x = std::max(c.blocks.x, blocks.x);
    result.blocks.y = std::max(c.blocks.y, blocks.y);
    result.blocks.z = std::max(c.blocks.z, blocks.z);

    result.threads.x = std::max(c.threads.x, threads.x);
    result.threads.y = std::max(c.threads.y, threads.y);
    result.threads.z = std::max(c.threads.z, threads.z);

    result.min_threads.x = std::max(c.min_threads.x, min_threads.x);
    result.min_threads.y = std::max(c.min_threads.y, min_threads.y);
    result.min_threads.z = std::max(c.min_threads.z, min_threads.z);

    return result;
  }

  RAJA_INLINE
  int num_blocks() const {
    if(blocks.x == 0 && blocks.y == 0 && blocks.z == 0){
      return 0;
    }
    return (blocks.x ? blocks.x : 1) *
           (blocks.y ? blocks.y : 1) *
           (blocks.z ? blocks.z : 1);
  }

  RAJA_INLINE
  int num_threads() const {
    if(threads.x == 0 && threads.y == 0 && threads.z == 0){
      return 0;
    }
    return (threads.x ? threads.x : 1) *
           (threads.y ? threads.y : 1) *
           (threads.z ? threads.z : 1);
  }


  RAJA_INLINE
  void clamp_to_min_threads() {
    threads.x = std::max(min_threads.x, threads.x);
    threads.y = std::max(min_threads.y, threads.y);
    threads.z = std::max(min_threads.z, threads.z);
  };

};







template <camp::idx_t cur_stmt, camp::idx_t num_stmts, typename StmtList>
struct CudaStatementListExecutorHelper {

  using next_helper_t =
      CudaStatementListExecutorHelper<cur_stmt + 1, num_stmts, StmtList>;

  using cur_stmt_t = camp::at_v<StmtList, cur_stmt>;

  template <typename Data>
  inline static RAJA_DEVICE void exec(Data &data, bool thread_active)
  {
    // Execute stmt
    cur_stmt_t::exec(data, thread_active);

    // Execute next stmt
    next_helper_t::exec(data, thread_active);
  }


  template <typename Data>
  inline static LaunchDims calculateDimensions(Data &data)
  {
    // Compute this statements launch dimensions
    LaunchDims statement_dims = cur_stmt_t::calculateDimensions(data);

    // call the next statement in the list
    LaunchDims next_dims = next_helper_t::calculateDimensions(data);

    // Return the maximum of the two
    return statement_dims.max(next_dims);
  }
};

template <camp::idx_t num_stmts, typename StmtList>
struct CudaStatementListExecutorHelper<num_stmts, num_stmts, StmtList> {

  template <typename Data>
  inline static RAJA_DEVICE void exec(Data &, bool)
  {
    // nop terminator
  }

  template <typename Data>
  inline static LaunchDims calculateDimensions(Data &)
  {
    return LaunchDims();
  }
};


template <typename Data, typename Policy>
struct CudaStatementExecutor;

template <typename Data, typename StmtList>
struct CudaStatementListExecutor;


template <typename Data, typename... Stmts>
struct CudaStatementListExecutor<Data, StatementList<Stmts...>> {

  using enclosed_stmts_t =
      camp::list<CudaStatementExecutor<Data, Stmts>...>;

  static constexpr size_t num_stmts = sizeof...(Stmts);

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    // Execute statements in order with helper class
    CudaStatementListExecutorHelper<0, num_stmts, enclosed_stmts_t>::exec(data, thread_active);
  }



  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    // Compute this statements launch dimensions
    return CudaStatementListExecutorHelper<0, num_stmts, enclosed_stmts_t>::
        calculateDimensions(data);
  }
};


template <typename StmtList, typename Data>
using cuda_statement_list_executor_t = CudaStatementListExecutor<
    Data,
    StmtList>;



}  // namespace internal
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
