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
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#ifndef RAJA_policy_hip_kernel_internal_HPP
#define RAJA_policy_hip_kernel_internal_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include <cassert>
#include <climits>

#include "camp/camp.hpp"

#include "RAJA/pattern/kernel.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/hip/MemUtils_HIP.hpp"
#include "RAJA/policy/hip/policy.hpp"


namespace RAJA
{


/*!
 * Policy for For<>, executes loop iteration by distributing them over threads.
 * This does no (additional) work-sharing between thread blocks.
 */

struct hip_thread_exec : public RAJA::make_policy_pattern_launch_platform_t<
                              RAJA::Policy::hip,
                              RAJA::Pattern::forall,
                              RAJA::Launch::undefined,
                              RAJA::Platform::hip> {
};


/*!
 * Policy for For<>, executes loop iteration by distributing iterations
 * exclusively over blocks.
 */

struct hip_block_exec : public RAJA::make_policy_pattern_launch_platform_t<
                             RAJA::Policy::hip,
                             RAJA::Pattern::forall,
                             RAJA::Launch::undefined,
                             RAJA::Platform::hip> {
};


/*!
 * Policy for For<>, executes loop iteration by distributing work over
 * physical blocks and executing sequentially within blocks.
 */

template <size_t num_blocks>
struct hip_block_seq_exec : public RAJA::make_policy_pattern_launch_platform_t<
                                 RAJA::Policy::hip,
                                 RAJA::Pattern::forall,
                                 RAJA::Launch::undefined,
                                 RAJA::Platform::hip> {
};


/*!
 * Policy for For<>, executes loop iteration by distributing them over threads
 * and blocks, but limiting the number of threads to num_threads.
 */
template <size_t num_threads>
struct hip_threadblock_exec
    : public RAJA::make_policy_pattern_launch_platform_t<
          RAJA::Policy::hip,
          RAJA::Pattern::forall,
          RAJA::Launch::undefined,
          RAJA::Platform::hip> {
};


namespace internal
{

RAJA_INLINE
int get_size(hip_dim_t dims)
{
  if(dims.x == 0 && dims.y == 0 && dims.z == 0){
    return 0;
  }
  return (dims.x ? dims.x : 1) *
         (dims.y ? dims.y : 1) *
         (dims.z ? dims.z : 1);
}

struct LaunchDims {

  hip_dim_t blocks;
  hip_dim_t min_blocks;
  hip_dim_t threads;
  hip_dim_t min_threads;

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
  int num_blocks() const {
    return get_size(blocks);
  }

  RAJA_INLINE
  int num_threads() const {
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


struct HipFixedMaxBlocksData
{
  int multiProcessorCount;
  int maxThreadsPerMultiProcessor;
};

RAJA_INLINE
int hip_max_blocks(int block_size)
{
  static HipFixedMaxBlocksData data = {-1, -1};

  if (data.multiProcessorCount < 0) {
    hipDeviceProp_t& prop = hip::device_prop();
    data.multiProcessorCount = prop.multiProcessorCount;
    data.maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
  }

  int max_blocks = data.multiProcessorCount *
                  (data.maxThreadsPerMultiProcessor / block_size);

  // printf("MAX_BLOCKS=%d\n", max_blocks);

  return max_blocks;
}

struct HipOccMaxBlocksThreadsData
{
  int prev_shmem_size;
  int max_blocks;
  int max_threads;
};

template < typename RAJA_UNUSED_ARG(UniqueMarker), typename Func >
RAJA_INLINE
void hip_occupancy_max_blocks_threads(Func&& func, int shmem_size,
                                       int &max_blocks, int &max_threads)
{
  static HipOccMaxBlocksThreadsData data = {-1, -1, -1};

  if (data.prev_shmem_size != shmem_size) {

#ifdef RAJA_ENABLE_HIP_OCCUPANCY_CALCULATOR
    hipErrchk(hipOccupancyMaxPotentialBlockSize(
        &data.max_blocks, &data.max_threads, func, shmem_size));
#else
    RAJA_UNUSED_VAR(func);
    data.max_blocks = 64;
    data.max_threads = 1024;
#endif

    data.prev_shmem_size = shmem_size;

  }

  max_blocks  = data.max_blocks;
  max_threads = data.max_threads;

}

struct HipOccMaxBlocksFixedThreadsData
{
  int prev_shmem_size;
  int max_blocks;
  int multiProcessorCount;
};

template < typename RAJA_UNUSED_ARG(UniqueMarker), int num_threads, typename Func >
RAJA_INLINE
void hip_occupancy_max_blocks(Func&& func, int shmem_size,
                               int &max_blocks)
{
  static HipOccMaxBlocksFixedThreadsData data = {-1, -1, -1};

  if (data.prev_shmem_size != shmem_size) {

#ifdef RAJA_ENABLE_HIP_OCCUPANCY_CALCULATOR
    hipErrchk(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &data.max_blocks, func, num_threads, shmem_size));
#else
    RAJA_UNUSED_VAR(func);
    data.max_blocks = 2;
#endif

    if (data.multiProcessorCount < 0) {

      data.multiProcessorCount = hip::device_prop().multiProcessorCount;

    }

    data.max_blocks *= data.multiProcessorCount;

    data.prev_shmem_size = shmem_size;

  }

  max_blocks = data.max_blocks;

}

struct HipOccMaxBlocksVariableThreadsData
{
  int prev_shmem_size;
  int prev_num_threads;
  int max_blocks;
  int multiProcessorCount;
};

template < typename RAJA_UNUSED_ARG(UniqueMarker), typename Func >
RAJA_INLINE
void hip_occupancy_max_blocks(Func&& func, int shmem_size,
                               int &max_blocks, int num_threads)
{
  static HipOccMaxBlocksVariableThreadsData data = {-1, -1, -1, -1};

  if ( data.prev_shmem_size  != shmem_size ||
       data.prev_num_threads != num_threads ) {

#ifdef RAJA_ENABLE_HIP_OCCUPANCY_CALCULATOR
    hipErrchk(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &data.max_blocks, func, num_threads, shmem_size));
#else
    RAJA_UNUSED_VAR(func);
    data.max_blocks = 2;
#endif

    if (data.multiProcessorCount < 0) {

      data.multiProcessorCount = hip::device_prop().multiProcessorCount;

    }

    data.max_blocks *= data.multiProcessorCount;

    data.prev_shmem_size  = shmem_size;
    data.prev_num_threads = num_threads;

  }

  max_blocks = data.max_blocks;

}



template <camp::idx_t cur_stmt, camp::idx_t num_stmts, typename StmtList>
struct HipStatementListExecutorHelper {

  using next_helper_t =
      HipStatementListExecutorHelper<cur_stmt + 1, num_stmts, StmtList>;

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
struct HipStatementListExecutorHelper<num_stmts, num_stmts, StmtList> {

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
struct HipStatementExecutor;

template <typename Data, typename StmtList, typename Types>
struct HipStatementListExecutor;


template <typename Data, typename... Stmts, typename Types>
struct HipStatementListExecutor<Data, StatementList<Stmts...>, Types> {

  using enclosed_stmts_t =
      camp::list<HipStatementExecutor<Data, Stmts, Types>...>;

  static constexpr size_t num_stmts = sizeof...(Stmts);

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    // Execute statements in order with helper class
    HipStatementListExecutorHelper<0, num_stmts, enclosed_stmts_t>::exec(data, thread_active);
  }



  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    // Compute this statements launch dimensions
    return HipStatementListExecutorHelper<0, num_stmts, enclosed_stmts_t>::
        calculateDimensions(data);
  }
};


template <typename StmtList, typename Data, typename Types>
using hip_statement_list_executor_t = HipStatementListExecutor<
    Data,
    StmtList,
    Types>;



}  // namespace internal
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_HIP guard

#endif  // closing endif for header file include guard
