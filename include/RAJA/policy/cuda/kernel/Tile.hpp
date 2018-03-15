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


#ifndef RAJA_policy_cuda_kernel_Tile_HPP
#define RAJA_policy_cuda_kernel_Tile_HPP

#include "RAJA/config.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"

#ifdef RAJA_ENABLE_CUDA

#include "RAJA/pattern/kernel/Tile.hpp"
#include "RAJA/pattern/kernel/internal.hpp"

#include "camp/camp.hpp"
#include "camp/concepts.hpp"
#include "camp/tuple.hpp"

#include <iostream>
#include <type_traits>

namespace RAJA
{
namespace internal
{


template <typename Data,
          camp::idx_t ArgumentId,
          typename TPol,
          typename... EnclosedStmts,
          typename IndexCalc>
struct CudaStatementExecutor<Data,
                             statement::Tile<ArgumentId,
                                             TPol,
                                             seq_exec,
                                             EnclosedStmts...>,
                             IndexCalc> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  using enclosed_stmts_t =
      CudaStatementListExecutor<Data, stmt_list_t, IndexCalc>;
  enclosed_stmts_t enclosed_stmts;

  inline __device__ void exec(Data &data,
                              int num_logical_blocks,
                              int block_carry)
  {
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
      enclosed_stmts.exec(data, num_logical_blocks, block_carry);
    }


    // Set range back to original values
    segment = orig_segment;
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

    // privatize data, so we can mess with the segments
    using data_t = camp::decay<Data>;
    data_t private_data = data;

    // Get original segment
    auto &segment = camp::get<ArgumentId>(private_data.segment_tuple);

    // restrict to first tile
    segment = segment.slice(0, TPol::chunk_size);

    // compute dimensions of children with segment restricted to tile
    LaunchDim dim =
        enclosed_stmts.calculateDimensions(private_data, max_physical);


    return dim;
  }
};


}  // end namespace internal
}  // end namespace RAJA

#endif  // RAJA_ENABLE_CUDA
#endif  /* RAJA_pattern_kernel_HPP */
