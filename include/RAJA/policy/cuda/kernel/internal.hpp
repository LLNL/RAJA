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
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
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


namespace RAJA
{


namespace internal
{

RAJA_INLINE
size_t get_size(cuda_dim_t dims)
{
  if(dims.x == 0 && dims.y == 0 && dims.z == 0){
    return 0;
  }
  return size_t(dims.x ? dims.x : 1) *
         size_t(dims.y ? dims.y : 1) *
         size_t(dims.z ? dims.z : 1);
}

struct LaunchDims {

  cuda_dim_t blocks;
  cuda_dim_t min_blocks;
  cuda_dim_t threads;
  cuda_dim_t min_threads;

  RAJA_INLINE
  RAJA_HOST_DEVICE
  LaunchDims() : blocks{0,0,0},  min_blocks{0,0,0},
                 threads{0,0,0}, min_threads{0,0,0} {}


  RAJA_INLINE
  RAJA_HOST_DEVICE
  LaunchDims(LaunchDims const &c) :
  blocks(c.blocks),   min_blocks(c.min_blocks),
  threads(c.threads), min_threads(c.min_threads)
  {
  }

  RAJA_INLINE
  LaunchDims max(LaunchDims const &c) const
  {
    LaunchDims result;

    result.blocks.x = std::max(c.blocks.x, blocks.x);
    result.blocks.y = std::max(c.blocks.y, blocks.y);
    result.blocks.z = std::max(c.blocks.z, blocks.z);

    result.min_blocks.x = std::max(c.min_blocks.x, min_blocks.x);
    result.min_blocks.y = std::max(c.min_blocks.y, min_blocks.y);
    result.min_blocks.z = std::max(c.min_blocks.z, min_blocks.z);

    result.threads.x = std::max(c.threads.x, threads.x);
    result.threads.y = std::max(c.threads.y, threads.y);
    result.threads.z = std::max(c.threads.z, threads.z);

    result.min_threads.x = std::max(c.min_threads.x, min_threads.x);
    result.min_threads.y = std::max(c.min_threads.y, min_threads.y);
    result.min_threads.z = std::max(c.min_threads.z, min_threads.z);

    return result;
  }

  RAJA_INLINE
  size_t num_blocks() const {
    return get_size(blocks);
  }

  RAJA_INLINE
  size_t num_threads() const {
    return get_size(threads);
  }


  RAJA_INLINE
  void clamp_to_min_blocks() {
    blocks.x = std::max(min_blocks.x, blocks.x);
    blocks.y = std::max(min_blocks.y, blocks.y);
    blocks.z = std::max(min_blocks.z, blocks.z);
  };

  RAJA_INLINE
  void clamp_to_min_threads() {
    threads.x = std::max(min_threads.x, threads.x);
    threads.y = std::max(min_threads.y, threads.y);
    threads.z = std::max(min_threads.z, threads.z);
  };

};


struct CudaFixedMaxBlocksData
{
  int multiProcessorCount;
  int maxThreadsPerMultiProcessor;
};

RAJA_INLINE
size_t cuda_max_blocks(size_t block_size)
{
  static CudaFixedMaxBlocksData data = {-1, -1};

  if (data.multiProcessorCount < 0) {
    cudaDeviceProp& prop = cuda::device_prop();
    data.multiProcessorCount = prop.multiProcessorCount;
    data.maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
  }

  size_t max_blocks = data.multiProcessorCount *
                  (data.maxThreadsPerMultiProcessor / block_size);

  // printf("MAX_BLOCKS=%d\n", max_blocks);

  return max_blocks;
}

struct CudaOccMaxBlocksThreadsData
{
  int prev_shmem_size;
  int max_blocks;
  int max_threads;
};

template < typename RAJA_UNUSED_ARG(UniqueMarker), typename Func >
RAJA_INLINE
void cuda_occupancy_max_blocks_threads(Func&& func, int shmem_size,
                                       size_t &max_blocks, size_t &max_threads)
{
  static CudaOccMaxBlocksThreadsData data = {-1, -1, -1};

  if (data.prev_shmem_size != shmem_size) {

    cudaErrchk(cudaOccupancyMaxPotentialBlockSize(
        &data.max_blocks, &data.max_threads, func, shmem_size));

    data.prev_shmem_size = shmem_size;

  }

  max_blocks  = data.max_blocks;
  max_threads = data.max_threads;

}

struct CudaOccMaxBlocksFixedThreadsData
{
  int prev_shmem_size;
  int max_blocks;
  int multiProcessorCount;
};

template < typename RAJA_UNUSED_ARG(UniqueMarker), size_t num_threads, typename Func >
RAJA_INLINE
void cuda_occupancy_max_blocks(Func&& func, int shmem_size,
                               size_t &max_blocks)
{
  static CudaOccMaxBlocksFixedThreadsData data = {-1, -1, -1};

  if (data.prev_shmem_size != shmem_size) {

    cudaErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &data.max_blocks, func, num_threads, shmem_size));

    if (data.multiProcessorCount < 0) {

      data.multiProcessorCount = cuda::device_prop().multiProcessorCount;

    }

    data.max_blocks *= data.multiProcessorCount;

    data.prev_shmem_size = shmem_size;

  }

  max_blocks = data.max_blocks;

}

struct CudaOccMaxBlocksVariableThreadsData
{
  int prev_shmem_size;
  size_t prev_num_threads;
  size_t max_blocks;
  int multiProcessorCount;
};

template < typename RAJA_UNUSED_ARG(UniqueMarker), typename Func >
RAJA_INLINE
void cuda_occupancy_max_blocks(Func&& func, int shmem_size,
                               size_t &max_blocks, size_t num_threads)
{
  static CudaOccMaxBlocksVariableThreadsData data = {0, 0, 0, 0};

  if ( data.prev_shmem_size  != shmem_size ||
       data.prev_num_threads != num_threads ) {

    int max_blocks(0);
    cudaErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_blocks, func, static_cast<int>(num_threads), shmem_size));

    data.max_blocks = max_blocks;

    if (data.multiProcessorCount < 0) {

      data.multiProcessorCount = cuda::device_prop().multiProcessorCount;

    }

    data.max_blocks *= data.multiProcessorCount;

    data.prev_shmem_size  = shmem_size;
    data.prev_num_threads = num_threads;

  }

  max_blocks = data.max_blocks;

}



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


template <typename Data, typename Policy, typename Types>
struct CudaStatementExecutor;

template <typename Data, typename StmtList, typename Types>
struct CudaStatementListExecutor;


template <typename Data, typename... Stmts, typename Types>
struct CudaStatementListExecutor<Data, StatementList<Stmts...>, Types> {

  using enclosed_stmts_t =
      camp::list<CudaStatementExecutor<Data, Stmts, Types>...>;

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


template <typename StmtList, typename Data, typename Types>
using cuda_statement_list_executor_t = CudaStatementListExecutor<
    Data,
    StmtList,
    Types>;



}  // namespace internal
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
