 /*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for CUDA tiled executors.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_cuda_kernel_Tile_HPP
#define RAJA_policy_cuda_kernel_Tile_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include <iostream>
#include <type_traits>

#include "camp/camp.hpp"
#include "camp/concepts.hpp"
#include "camp/tuple.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/kernel/Tile.hpp"
#include "RAJA/pattern/kernel/internal.hpp"

namespace RAJA
{
namespace internal
{

/*!
 * A specialized RAJA::kernel cuda_impl executor for statement::Tile
 * Assigns the tile segment to segment ArgumentId
 *
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename TPol,
          typename... EnclosedStmts,
          typename Types>
struct CudaStatementExecutor<
    Data,
    statement::Tile<ArgumentId, TPol, seq_exec, EnclosedStmts...>, Types>
{

  using stmt_list_t = StatementList<EnclosedStmts...>;
  using enclosed_stmts_t = CudaStatementListExecutor<Data, stmt_list_t, Types>;

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active){
    // Get the segment referenced by this Tile statement
    auto &segment = camp::get<ArgumentId>(data.segment_tuple);

    // Keep copy of original segment, so we can restore it
    using segment_t = camp::decay<decltype(segment)>;
    segment_t orig_segment = segment;

    using diff_type = segment_diff_type<ArgumentId, Data>;
    diff_type chunk_size = TPol::chunk_size;

    // compute trip count
    diff_type len = segment.end() - segment.begin();

    // Iterate through tiles
    for (diff_type i = 0; i < len; i += chunk_size) {

      // Assign our new tiled segment
      segment = orig_segment.slice(i, chunk_size);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active);
    }

    // Set range back to original values
    segment = orig_segment;
  }


  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {

    // privatize data, so we can mess with the segments
    using data_t = camp::decay<Data>;
    data_t private_data = data;

    // Get original segment
    auto &segment = camp::get<ArgumentId>(private_data.segment_tuple);

    // restrict to first tile
    segment = segment.slice(0, TPol::chunk_size);

    // compute dimensions of children with segment restricted to tile
    LaunchDims enclosed_dims =
        enclosed_stmts_t::calculateDimensions(private_data);

    return enclosed_dims;
  }
};


/*!
 * A specialized RAJA::kernel cuda_impl executor for statement::Tile
 * Assigns the tile segment to segment ArgumentId
 *
 */
template <typename Data,
          camp::idx_t ArgumentId,
          camp::idx_t chunk_size,
          int BlockDim,
          typename... EnclosedStmts,
          typename Types>
struct CudaStatementExecutor<
    Data,
    statement::Tile<ArgumentId,
                    RAJA::tile_fixed<chunk_size>,
                    cuda_block_xyz_direct<BlockDim>,
                    EnclosedStmts...>,
                    Types>
  {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  using enclosed_stmts_t = CudaStatementListExecutor<Data, stmt_list_t, Types>;

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    // Get the segment referenced by this Tile statement
    auto &segment = camp::get<ArgumentId>(data.segment_tuple);

    using segment_t = camp::decay<decltype(segment)>;

    // compute trip count
    using diff_type = segment_diff_type<ArgumentId, Data>;
    diff_type len = segment.end() - segment.begin();
    diff_type i = get_cuda_dim<BlockDim>(blockIdx) * chunk_size;

    // check have chunk
    if (i < len) {

      // Keep copy of original segment, so we can restore it
      segment_t orig_segment = segment;

      // Assign our new tiled segment
      segment = orig_segment.slice(i, chunk_size);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active);

      // Set range back to original values
      segment = orig_segment;
    }
  }


  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {

    // Compute how many blocks
    using diff_type = segment_diff_type<ArgumentId, Data>;
    diff_type len = segment_length<ArgumentId>(data);
    diff_type num_blocks = len / chunk_size;
    if (num_blocks * chunk_size < len) {
      num_blocks++;
    }

    LaunchDims dims;
    set_cuda_dim<BlockDim>(dims.blocks, num_blocks);

    // since we are direct-mapping, we REQUIRE len
    set_cuda_dim<BlockDim>(dims.min_blocks, num_blocks);


    // privatize data, so we can mess with the segments
    using data_t = camp::decay<Data>;
    data_t private_data = data;

    // Get original segment
    auto &segment = camp::get<ArgumentId>(private_data.segment_tuple);

    // restrict to first tile
    segment = segment.slice(0, chunk_size);


    LaunchDims enclosed_dims =
        enclosed_stmts_t::calculateDimensions(private_data);

    return dims.max(enclosed_dims);
  }
};

/*!
 * A specialized RAJA::kernel cuda_impl executor for statement::Tile
 * Assigns the tile segment to segment ArgumentId
 *
 */
template <typename Data,
          camp::idx_t ArgumentId,
          camp::idx_t chunk_size,
          int BlockDim,
          typename... EnclosedStmts,
          typename Types>
struct CudaStatementExecutor<
    Data,
    statement::Tile<ArgumentId,
                    RAJA::tile_fixed<chunk_size>,
                    cuda_block_xyz_loop<BlockDim>,
                    EnclosedStmts...>, Types>
  {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  using enclosed_stmts_t = CudaStatementListExecutor<Data, stmt_list_t, Types>;

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    // Get the segment referenced by this Tile statement
    auto &segment = camp::get<ArgumentId>(data.segment_tuple);

    // Keep copy of original segment, so we can restore it
    using segment_t = camp::decay<decltype(segment)>;
    segment_t orig_segment = segment;

    // compute trip count
    using diff_type = segment_diff_type<ArgumentId, Data>;
    diff_type len = segment.end() - segment.begin();
    diff_type i0 = get_cuda_dim<BlockDim>(blockIdx) * chunk_size;
    diff_type i_stride = get_cuda_dim<BlockDim>(gridDim) * chunk_size;

    // Iterate through grid stride of chunks
    for (diff_type i = i0; i < len; i += i_stride) {

      // Assign our new tiled segment
      segment = orig_segment.slice(i, chunk_size);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active);
    }

    // Set range back to original values
    segment = orig_segment;
  }


  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {

    // Compute how many blocks
    using diff_type = segment_diff_type<ArgumentId, Data>;
    diff_type len = segment_length<ArgumentId>(data);
    diff_type num_blocks = len / chunk_size;
    if (num_blocks * chunk_size < len) {
      num_blocks++;
    }

    LaunchDims dims;
    set_cuda_dim<BlockDim>(dims.blocks, num_blocks);



    // privatize data, so we can mess with the segments
    using data_t = camp::decay<Data>;
    data_t private_data = data;

    // Get original segment
    auto &segment = camp::get<ArgumentId>(private_data.segment_tuple);

    // restrict to first tile
    segment = segment.slice(0, chunk_size);


    LaunchDims enclosed_dims =
        enclosed_stmts_t::calculateDimensions(private_data);

    return dims.max(enclosed_dims);
  }
};



/*!
 * A specialized RAJA::kernel cuda_impl executor for statement::Tile
 * Assigns the tile segment to segment ArgumentId
 *
 */
template <typename Data,
          camp::idx_t ArgumentId,
          camp::idx_t chunk_size,
          int ThreadDim,
          typename ... EnclosedStmts,
          typename Types>
struct CudaStatementExecutor<
  Data,
  statement::Tile<ArgumentId,
                  RAJA::tile_fixed<chunk_size>,
                  cuda_thread_xyz_direct<ThreadDim>,
                  EnclosedStmts ...>, Types>{

  using stmt_list_t = StatementList<EnclosedStmts ...>;

  using enclosed_stmts_t = CudaStatementListExecutor<Data, stmt_list_t, Types>;

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    // Get the segment referenced by this Tile statement
    auto &segment = camp::get<ArgumentId>(data.segment_tuple);

    // Keep copy of original segment, so we can restore it
    using segment_t = camp::decay<decltype(segment)>;
    segment_t orig_segment = segment;

    // compute trip count
    using diff_type = segment_diff_type<ArgumentId, Data>;
    diff_type len = segment.end() - segment.begin();
    diff_type i = get_cuda_dim<ThreadDim>(threadIdx) * chunk_size;

    // execute enclosed statements if any thread will
    // but mask off threads without work
    bool have_work = i < len;

    // Assign our new tiled segment
    diff_type slice_size = have_work ? chunk_size : 0;
    segment = orig_segment.slice(i, slice_size);

    // execute enclosed statements
    enclosed_stmts_t::exec(data, thread_active && have_work);

    // Set range back to original values
    segment = orig_segment;
  }


  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {

    // Compute how many blocks
    using diff_type = segment_diff_type<ArgumentId, Data>;
    diff_type len = segment_length<ArgumentId>(data);
    diff_type num_threads = len / chunk_size;
    if(num_threads * chunk_size < len){
      num_threads++;
    }

    LaunchDims dims;
    set_cuda_dim<ThreadDim>(dims.threads, num_threads);
    set_cuda_dim<ThreadDim>(dims.min_threads, num_threads);

    // privatize data, so we can mess with the segments
    using data_t = camp::decay<Data>;
    data_t private_data = data;

    // Get original segment
    auto &segment = camp::get<ArgumentId>(private_data.segment_tuple);

    // restrict to first tile
    segment = segment.slice(0, chunk_size);


    LaunchDims enclosed_dims =
      enclosed_stmts_t::calculateDimensions(private_data);

    return(dims.max(enclosed_dims));
  }
};


/*!
 * A specialized RAJA::kernel cuda_impl executor for statement::Tile
 * Assigns the tile segment to segment ArgumentId
 *
 */
template <typename Data,
          camp::idx_t ArgumentId,
          camp::idx_t chunk_size,
          int ThreadDim,
          int MinThreads,
          typename ... EnclosedStmts,
          typename Types>
struct CudaStatementExecutor<
  Data,
  statement::Tile<ArgumentId,
                  RAJA::tile_fixed<chunk_size>,
                  cuda_thread_xyz_loop<ThreadDim, MinThreads>,
                  EnclosedStmts ...>, Types>{

  using stmt_list_t = StatementList<EnclosedStmts ...>;

  using enclosed_stmts_t = CudaStatementListExecutor<Data, stmt_list_t, Types>;

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    // Get the segment referenced by this Tile statement
    auto &segment = camp::get<ArgumentId>(data.segment_tuple);

    // Keep copy of original segment, so we can restore it
    using segment_t = camp::decay<decltype(segment)>;
    segment_t orig_segment = segment;

    // compute trip count
    using diff_type = segment_diff_type<ArgumentId, Data>;
    diff_type len = segment_length<ArgumentId>(data);
    diff_type i0 = get_cuda_dim<ThreadDim>(threadIdx) * chunk_size;

    // Get our stride from the dimension
    diff_type i_stride = get_cuda_dim<ThreadDim>(blockDim) * chunk_size;

    // Iterate through grid stride of chunks
    for (diff_type ii = 0; ii < len; ii += i_stride) {
      diff_type i = ii + i0;

      // execute enclosed statements if any thread will
      // but mask off threads without work
      bool have_work = i < len;

      // Assign our new tiled segment
      diff_type slice_size = have_work ? chunk_size : 0;
      segment = orig_segment.slice(i, slice_size);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active && have_work);
    }

    // Set range back to original values
    segment = orig_segment;
  }


  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {

    // Compute how many blocks
    using diff_type = segment_diff_type<ArgumentId, Data>;
    diff_type len = segment_length<ArgumentId>(data);
    diff_type num_threads = len / chunk_size;
    if(num_threads * chunk_size < len){
      num_threads++;
    }
    num_threads = std::max(num_threads, (diff_type)MinThreads);

    LaunchDims dims;
    set_cuda_dim<ThreadDim>(dims.threads, num_threads);
    set_cuda_dim<ThreadDim>(dims.min_threads, MinThreads);

    // privatize data, so we can mess with the segments
    using data_t = camp::decay<Data>;
    data_t private_data = data;

    // Get original segment
    auto &segment = camp::get<ArgumentId>(private_data.segment_tuple);

    // restrict to first tile
    segment = segment.slice(0, chunk_size);


    LaunchDims enclosed_dims =
      enclosed_stmts_t::calculateDimensions(private_data);

    return(dims.max(enclosed_dims));
  }
};




}  // end namespace internal
}  // end namespace RAJA

#endif  // RAJA_ENABLE_CUDA
#endif  /* RAJA_pattern_kernel_HPP */
