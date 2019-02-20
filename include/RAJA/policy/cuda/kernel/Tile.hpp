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
          typename... EnclosedStmts>
struct CudaStatementExecutor<
    Data,
    statement::Tile<ArgumentId, TPol, seq_exec, EnclosedStmts...>>
{

  using stmt_list_t = StatementList<EnclosedStmts...>;
  using enclosed_stmts_t = CudaStatementListExecutor<Data, stmt_list_t>;

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active){
    // Get the segment referenced by this Tile statement
    auto &segment = camp::get<ArgumentId>(data.segment_tuple);

    // Keep copy of original segment, so we can restore it
    using segment_t = camp::decay<decltype(segment)>;
    segment_t orig_segment = segment;

    int chunk_size = TPol::chunk_size;

    // compute trip count
    int len = segment.end() - segment.begin();

    // Iterate through tiles
    for (int i = 0; i < len; i += chunk_size) {

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
          typename... EnclosedStmts>
struct CudaStatementExecutor<
    Data,
    statement::Tile<ArgumentId,
                    RAJA::statement::tile_fixed<chunk_size>,
                    cuda_block_xyz_loop<BlockDim>,
                    EnclosedStmts...>>
  {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  using enclosed_stmts_t = CudaStatementListExecutor<Data, stmt_list_t>;

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
    int len = segment.end() - segment.begin();
    auto i0 = get_cuda_dim<BlockDim>(blockIdx) * chunk_size;
    auto i_stride = get_cuda_dim<BlockDim>(gridDim) * chunk_size;

    // Iterate through grid stride of chunks
    for (int i = i0; i < len; i += i_stride) {

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
    int len = segment_length<ArgumentId>(data);
    int num_blocks = len / chunk_size;
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


}  // end namespace internal
}  // end namespace RAJA

#endif  // RAJA_ENABLE_CUDA
#endif  /* RAJA_pattern_kernel_HPP */
