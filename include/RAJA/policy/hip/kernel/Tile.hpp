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


#ifndef RAJA_policy_hip_kernel_Tile_HPP
#define RAJA_policy_hip_kernel_Tile_HPP

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
 * A specialized RAJA::kernel hip_impl executor for statement::Tile
 * Assigns the tile segment to segment ArgumentId
 * Meets all sync requirements
 */
template <typename Data,
          camp::idx_t ArgumentId,
          camp::idx_t chunk_size,
          typename IndexMapper,
          kernel_sync_requirement sync,
          typename... EnclosedStmts,
          typename Types>
struct HipStatementExecutor<
    Data,
    statement::Tile<ArgumentId,
                    RAJA::tile_fixed<chunk_size>,
                    RAJA::policy::hip::hip_indexer<iteration_mapping::Direct, sync, IndexMapper>,
                    EnclosedStmts...>,
                    Types>
  {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  using enclosed_stmts_t = HipStatementListExecutor<Data, stmt_list_t, Types>;

  using diff_t = segment_diff_type<ArgumentId, Data>;

  using DimensionCalculator = KernelDimensionCalculator<RAJA::policy::hip::hip_indexer<iteration_mapping::Direct, sync, IndexMapper>>;

  static inline RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    // Get the segment referenced by this Tile statement
    auto &segment = camp::get<ArgumentId>(data.segment_tuple);

    using segment_t = camp::decay<decltype(segment)>;

    // compute trip count
    const diff_t len = segment.end() - segment.begin();
    const diff_t i = IndexMapper::template index<diff_t>() * static_cast<diff_t>(chunk_size);

    // execute enclosed statements if any thread will
    // but mask off threads without work
    const bool have_work = (i < len);

    // Keep copy of original segment, so we can restore it
    segment_t orig_segment = segment;

    // Assign our new tiled segment
    segment = orig_segment.slice(i, static_cast<diff_t>(chunk_size));

    // execute enclosed statements
    enclosed_stmts_t::exec(data, thread_active && have_work);

    // Set range back to original values
    segment = orig_segment;
  }

  static inline
  LaunchDims calculateDimensions(Data const &data)
  {
    // Compute how many chunks
    const diff_t full_len = segment_length<ArgumentId>(data);
    const diff_t len = RAJA_DIVIDE_CEILING_INT(full_len, static_cast<diff_t>(chunk_size));

    HipDims my_dims(0), my_min_dims(0);
    DimensionCalculator{}.set_dimensions(my_dims, my_min_dims, len);
    LaunchDims dims{my_dims, my_min_dims};

    // privatize data, so we can mess with the segments
    using data_t = camp::decay<Data>;
    data_t private_data = data;

    // Get original segment
    auto &segment = camp::get<ArgumentId>(private_data.segment_tuple);

    // restrict to first tile
    segment = segment.slice(0, static_cast<diff_t>(chunk_size));

    LaunchDims enclosed_dims =
        enclosed_stmts_t::calculateDimensions(private_data);

    return dims.max(enclosed_dims);
  }
};

/*!
 * A specialized RAJA::kernel hip_impl executor for statement::Tile
 * Assigns the tile segment to segment ArgumentId
 * Meets all sync requirements
 */
template <typename Data,
          camp::idx_t ArgumentId,
          camp::idx_t chunk_size,
          typename IndexMapper,
          typename... EnclosedStmts,
          typename Types>
struct HipStatementExecutor<
    Data,
    statement::Tile<ArgumentId,
                    RAJA::tile_fixed<chunk_size>,
                    RAJA::policy::hip::hip_indexer<iteration_mapping::StridedLoop, kernel_sync_requirement::sync, IndexMapper>,
                    EnclosedStmts...>, Types>
  {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  using enclosed_stmts_t = HipStatementListExecutor<Data, stmt_list_t, Types>;

  using diff_t = segment_diff_type<ArgumentId, Data>;

  using DimensionCalculator = KernelDimensionCalculator<RAJA::policy::hip::hip_indexer<iteration_mapping::StridedLoop, kernel_sync_requirement::sync, IndexMapper>>;

  static inline RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    // Get the segment referenced by this Tile statement
    auto &segment = camp::get<ArgumentId>(data.segment_tuple);

    // Keep copy of original segment, so we can restore it
    using segment_t = camp::decay<decltype(segment)>;
    segment_t orig_segment = segment;

    // compute trip count
    const diff_t len = segment.end() - segment.begin();
    const diff_t i_init = IndexMapper::template index<diff_t>() * static_cast<diff_t>(chunk_size);
    const diff_t i_stride = IndexMapper::template size<diff_t>() * static_cast<diff_t>(chunk_size);

    // Iterate through in chunks
    // threads will have the same numbers of iterations
    for (diff_t ii = 0; ii < len; ii += i_stride) {
      const diff_t i = ii + i_init;

      // execute enclosed statements if any thread will
      // but mask off threads without work
      const bool have_work = (i < len);

      // Assign our new tiled segment
      segment = orig_segment.slice(i, static_cast<diff_t>(chunk_size));

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active && have_work);
    }

    // Set range back to original values
    segment = orig_segment;
  }

  static inline
  LaunchDims calculateDimensions(Data const &data)
  {
    // Compute how many chunks
    const diff_t full_len = segment_length<ArgumentId>(data);
    const diff_t len = RAJA_DIVIDE_CEILING_INT(full_len, static_cast<diff_t>(chunk_size));

    HipDims my_dims(0), my_min_dims(0);
    DimensionCalculator{}.set_dimensions(my_dims, my_min_dims, len);
    LaunchDims dims{my_dims, my_min_dims};

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
 * A specialized RAJA::kernel hip_impl executor for statement::Tile
 * Assigns the tile segment to segment ArgumentId
 * Meets no sync requirements
 */
template <typename Data,
          camp::idx_t ArgumentId,
          camp::idx_t chunk_size,
          typename IndexMapper,
          typename... EnclosedStmts,
          typename Types>
struct HipStatementExecutor<
    Data,
    statement::Tile<ArgumentId,
                    RAJA::tile_fixed<chunk_size>,
                    RAJA::policy::hip::hip_indexer<iteration_mapping::StridedLoop, kernel_sync_requirement::none, IndexMapper>,
                    EnclosedStmts...>, Types>
  {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  using enclosed_stmts_t = HipStatementListExecutor<Data, stmt_list_t, Types>;

  using diff_t = segment_diff_type<ArgumentId, Data>;

  using DimensionCalculator = KernelDimensionCalculator<RAJA::policy::hip::hip_indexer<iteration_mapping::StridedLoop, kernel_sync_requirement::none, IndexMapper>>;

  static inline RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    // Get the segment referenced by this Tile statement
    auto &segment = camp::get<ArgumentId>(data.segment_tuple);

    // Keep copy of original segment, so we can restore it
    using segment_t = camp::decay<decltype(segment)>;
    segment_t orig_segment = segment;

    // compute trip count
    const diff_t len = segment.end() - segment.begin();
    const diff_t i_init = IndexMapper::template index<diff_t>() * static_cast<diff_t>(chunk_size);
    const diff_t i_stride = IndexMapper::template size<diff_t>() * static_cast<diff_t>(chunk_size);

    // Iterate through one at a time
    // threads will have the different numbers of iterations
    for (diff_t i = i_init; i < len; i += i_stride) {

      // Assign our new tiled segment
      segment = orig_segment.slice(i, static_cast<diff_t>(chunk_size));

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active);
    }

    // Set range back to original values
    segment = orig_segment;
  }

  static inline
  LaunchDims calculateDimensions(Data const &data)
  {
    // Compute how many chunks
    const diff_t full_len = segment_length<ArgumentId>(data);
    const diff_t len = RAJA_DIVIDE_CEILING_INT(full_len, static_cast<diff_t>(chunk_size));

    HipDims my_dims(0), my_min_dims(0);
    DimensionCalculator{}.set_dimensions(my_dims, my_min_dims, len);
    LaunchDims dims{my_dims, my_min_dims};

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
 * A specialized RAJA::kernel hip_impl executor for statement::Tile
 * Assigns the tile segment to segment ArgumentId
 *
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename TPol,
          typename... EnclosedStmts,
          typename Types>
struct HipStatementExecutor<
    Data,
    statement::Tile<ArgumentId, TPol, seq_exec, EnclosedStmts...>, Types>
: HipStatementExecutor<Data, statement::Tile<ArgumentId, TPol,
    RAJA::policy::hip::hip_indexer<iteration_mapping::StridedLoop,
                                   kernel_sync_requirement::none,
                                   hip::IndexGlobal<named_dim::x, named_usage::ignored, named_usage::ignored>>,
    EnclosedStmts...>, Types>
{

};

}  // end namespace internal
}  // end namespace RAJA

#endif  // RAJA_ENABLE_HIP
#endif  /* RAJA_policy_hip_kernel_Tile_HPP */
