/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for HIP tiled executors.
 *
 ******************************************************************************
 */


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#ifndef RAJA_policy_hip_kernel_TileTCount_HPP
#define RAJA_policy_hip_kernel_TileTCount_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

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
 * A specialized RAJA::kernel hip_impl executor for statement::TileTCount
 * Assigns the tile segment to segment ArgumentId
 * Assigns the tile index to param ParamId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename ParamId,
          typename TPol,
          typename... EnclosedStmts,
          typename Types>
struct HipStatementExecutor<
    Data,
    statement::TileTCount<ArgumentId, ParamId, TPol, seq_exec, EnclosedStmts...>, Types>
    : public HipStatementExecutor<
        Data,
        statement::Tile<ArgumentId, TPol, seq_exec, EnclosedStmts...>, Types> {

  using Base = HipStatementExecutor<
      Data,
      statement::Tile<ArgumentId, TPol, seq_exec, EnclosedStmts...>, Types>;

  using typename Base::enclosed_stmts_t;
  using typename Base::diff_t;

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active){
    // Get the segment referenced by this Tile statement
    auto &segment = camp::get<ArgumentId>(data.segment_tuple);

    // Keep copy of original segment, so we can restore it
    using segment_t = camp::decay<decltype(segment)>;
    segment_t orig_segment = segment;

    diff_t chunk_size = TPol::chunk_size;

    // compute trip count
    diff_t len = segment.end() - segment.begin();

    // Iterate through tiles
    for (diff_t i = 0, t = 0; i < len; i += chunk_size, ++t) {

      // Assign our new tiled segment
      segment = orig_segment.slice(i, chunk_size);
      data.template assign_param<ParamId>(t);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active);
    }

    // Set range back to original values
    segment = orig_segment;
  }
};


/*!
 * A specialized RAJA::kernel hip_impl executor for statement::TileTCount
 * Assigns the tile segment to segment ArgumentId
 * Assigns the tile index to param ParamId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename ParamId,
          camp::idx_t chunk_size,
          typename Indexer,
          typename... EnclosedStmts,
          typename Types>
struct HipStatementExecutor<
    Data,
    statement::TileTCount<ArgumentId, ParamId,
                    RAJA::tile_fixed<chunk_size>,
                    RAJA::internal::HipIndexDirect<Indexer>,
                    EnclosedStmts...>,
                    Types>
    : public HipStatementExecutor<
        Data,
        statement::Tile<ArgumentId,
                        RAJA::tile_fixed<chunk_size>,
                        RAJA::internal::HipIndexDirect<Indexer>,
                        EnclosedStmts...>,
                        Types> {

  using Base = HipStatementExecutor<
      Data,
      statement::Tile<ArgumentId,
                      RAJA::tile_fixed<chunk_size>,
                      RAJA::internal::HipIndexDirect<Indexer>,
                      EnclosedStmts...>,
                      Types>;

  using typename Base::enclosed_stmts_t;
  using typename Base::diff_t;

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    // Get the segment referenced by this Tile statement
    auto &segment = camp::get<ArgumentId>(data.segment_tuple);

    using segment_t = camp::decay<decltype(segment)>;

    // compute trip count
    diff_t len = segment.end() - segment.begin();
    diff_t t = Indexer::template index<diff_t>();
    diff_t i = t * static_cast<diff_t>(chunk_size);

    // Keep copy of original segment, so we can restore it
    segment_t orig_segment = segment;

    // Assign our new tiled segment
    segment = orig_segment.slice(i, static_cast<diff_t>(chunk_size));
    data.template assign_param<ParamId>(t);

    // execute enclosed statements
    enclosed_stmts_t::exec(data, thread_active && (i < len));

    // Set range back to original values
    segment = orig_segment;
  }
};

/*!
 * A specialized RAJA::kernel hip_impl executor for statement::TileTCount
 * Assigns the tile segment to segment ArgumentId
 * Assigns the tile index to param ParamId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename ParamId,
          camp::idx_t chunk_size,
          typename Indexer,
          typename... EnclosedStmts,
          typename Types>
struct HipStatementExecutor<
    Data,
    statement::TileTCount<ArgumentId, ParamId,
                    RAJA::tile_fixed<chunk_size>,
                    RAJA::internal::HipIndexLoop<Indexer>,
                    EnclosedStmts...>,
                    Types>
    : public HipStatementExecutor<
        Data,
        statement::Tile<ArgumentId,
                        RAJA::tile_fixed<chunk_size>,
                        RAJA::internal::HipIndexLoop<Indexer>,
                        EnclosedStmts...>,
                        Types> {

  using Base = HipStatementExecutor<
      Data,
      statement::Tile<ArgumentId,
                      RAJA::tile_fixed<chunk_size>,
                      RAJA::internal::HipIndexLoop<Indexer>,
                      EnclosedStmts...>,
                      Types>;

  using typename Base::enclosed_stmts_t;
  using typename Base::diff_t;

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
    diff_t len = segment.end() - segment.begin();
    diff_t t_init = Indexer::template index<diff_t>();
    diff_t i_init = t_init * static_cast<diff_t>(chunk_size);
    diff_t t_stride = Indexer::template size<diff_t>();
    diff_t i_stride = t_stride * static_cast<diff_t>(chunk_size);

    // Iterate through of chunks
    for(diff_t ii = 0, t = t_init; ii < len; ii += i_stride, t += t_stride) {
      diff_t i = ii + i_init;

      // execute enclosed statements if any thread will
      // but mask off threads without work
      bool have_work = i < len;

      // Assign our new tiled segment
      segment = orig_segment.slice(i, static_cast<diff_t>(chunk_size));
      data.template assign_param<ParamId>(t);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active && have_work);
    }

    // Set range back to original values
    segment = orig_segment;
  }
};

}  // end namespace internal
}  // end namespace RAJA

#endif  // RAJA_ENABLE_HIP
#endif  /* RAJA_policy_hip_kernel_TileTCount_HPP */
