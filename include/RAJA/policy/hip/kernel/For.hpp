/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for HIP statement executors.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#ifndef RAJA_policy_hip_kernel_For_HPP
#define RAJA_policy_hip_kernel_For_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/hip/kernel/internal.hpp"


namespace RAJA
{

namespace internal
{

/*
 * Executor for thread work sharing loop inside HipKernel.
 * Mapping directly from a warp lane
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename... EnclosedStmts,
          typename Types>
struct HipStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::hip_warp_direct, EnclosedStmts...>,
    Types> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      HipStatementListExecutor<Data, stmt_list_t, NewTypes>;

  using diff_t = segment_diff_type<ArgumentId, Data>;


  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    diff_t len = segment_length<ArgumentId>(data);
    diff_t i = threadIdx.x;

    // assign thread id directly to offset
    data.template assign_offset<ArgumentId>(i);

    // execute enclosed statements if in bounds
    enclosed_stmts_t::exec(data, thread_active && (i<len));
  }


  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    // Get enclosed statements
    LaunchDims dims = enclosed_stmts_t::calculateDimensions(data);

    // we always get EXACTLY one warp by allocating one warp in the X dimension
    diff_t len = RAJA::policy::hip::WARP_SIZE;

    // request one thread per element in the segment
    set_hip_dim<0>(dims.threads, len);

    // since we are direct-mapping, we REQUIRE len
    set_hip_dim<0>(dims.min_threads, len);

    // is the lack of calculating enclosed dims here an error?
    return dims;
  }
};


/*
 * Executor for thread work sharing loop inside HipKernel.
 * Provides a warp-stride loop for each thread inside of a warp.
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename... EnclosedStmts,
          typename Types>
struct HipStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::hip_warp_loop, EnclosedStmts...>,
    Types> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      HipStatementListExecutor<Data, stmt_list_t, NewTypes>;

  using diff_t = segment_diff_type<ArgumentId, Data>;


  static
  inline RAJA_DEVICE void exec(Data &data, bool thread_active)
  {
    // block stride loop
    diff_t len = segment_length<ArgumentId>(data);
    diff_t i_init = threadIdx.x;
    diff_t i_stride = RAJA::policy::hip::WARP_SIZE;

    // Iterate through grid stride of chunks
    for (diff_t ii = 0; ii < len; ii += i_stride) {
      diff_t i = ii + i_init;

      // execute enclosed statements if any thread will
      // but mask off threads without work
      bool have_work = i < len;

      // Assign the x thread to the argument
      data.template assign_offset<ArgumentId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active && have_work);
    }
  }


  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    // Get enclosed statements
    LaunchDims dims = enclosed_stmts_t::calculateDimensions(data);

    // we always get EXACTLY one warp by allocating one warp in the X dimension
    diff_t len = RAJA::policy::hip::WARP_SIZE;

    // request one thread per element in the segment
    set_hip_dim<0>(dims.threads, len);

    // since we are direct-mapping, we REQUIRE len
    set_hip_dim<0>(dims.min_threads, len);

    return dims;
  }
};


/*
 * Executor for thread work sharing loop inside HipKernel.
 * Mapping directly from a warp lane
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename Mask,
          typename ... EnclosedStmts,
          typename Types>
struct HipStatementExecutor<
  Data,
  statement::For<ArgumentId, RAJA::hip_warp_masked_direct<Mask>,
                 EnclosedStmts ...>,
  Types> {

  using stmt_list_t = StatementList<EnclosedStmts ...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
          HipStatementListExecutor<Data, stmt_list_t, NewTypes>;

  using mask_t = Mask;

  using diff_t = segment_diff_type<ArgumentId, Data>;

  static_assert(mask_t::max_masked_size <= RAJA::policy::hip::WARP_SIZE,
                "BitMask is too large for HIP warp size");

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    diff_t len = segment_length<ArgumentId>(data);

    diff_t i = mask_t::maskValue((diff_t)threadIdx.x);

    // assign thread id directly to offset
    data.template assign_offset<ArgumentId>(i);

    // execute enclosed statements if in bounds
    enclosed_stmts_t::exec(data, thread_active && (i<len));
  }


  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    // Get enclosed statements
    LaunchDims dims = enclosed_stmts_t::calculateDimensions(data);

    // we always get EXACTLY one warp by allocating one warp in the X
    // dimension
    diff_t len = RAJA::policy::hip::WARP_SIZE;

    // request one thread per element in the segment
    set_hip_dim<0>(dims.threads, len);

    // since we are direct-mapping, we REQUIRE len
    set_hip_dim<0>(dims.min_threads, len);

    return(dims);
  }
};



/*
 * Executor for thread work sharing loop inside HipKernel.
 * Mapping directly from a warp lane
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename Mask,
          typename ... EnclosedStmts,
          typename Types>
struct HipStatementExecutor<
  Data,
  statement::For<ArgumentId, RAJA::hip_warp_masked_loop<Mask>,
                 EnclosedStmts ...>,
  Types> {

  using stmt_list_t = StatementList<EnclosedStmts ...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
          HipStatementListExecutor<Data, stmt_list_t, NewTypes>;

  using mask_t = Mask;

  using diff_t = segment_diff_type<ArgumentId, Data>;

  static_assert(mask_t::max_masked_size <= RAJA::policy::hip::WARP_SIZE,
                "BitMask is too large for HIP warp size");

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    // masked size strided loop
    diff_t len = segment_length<ArgumentId>(data);
    diff_t i_init = mask_t::maskValue((diff_t)threadIdx.x);
    diff_t i_stride = (diff_t) mask_t::max_masked_size;

    // Iterate through grid stride of chunks
    for (diff_t ii = 0; ii < len; ii += i_stride) {
      diff_t i = ii + i_init;

      // execute enclosed statements if any thread will
      // but mask off threads without work
      bool have_work = i < len;

      // Assign the x thread to the argument
      data.template assign_offset<ArgumentId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active && have_work);
    }
  }


  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    // Get enclosed statements
    LaunchDims dims = enclosed_stmts_t::calculateDimensions(data);

    // we always get EXACTLY one warp by allocating one warp in the X
    // dimension
    diff_t len = RAJA::policy::hip::WARP_SIZE;

    // request one thread per element in the segment
    set_hip_dim<0>(dims.threads, len);

    // since we are direct-mapping, we REQUIRE len
    set_hip_dim<0>(dims.min_threads, len);

    return(dims);
  }
};


/*
 * Executor for thread work sharing loop inside HipKernel.
 * Mapping directly from raw threadIdx.x
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename Mask,
          typename ... EnclosedStmts,
          typename Types>
struct HipStatementExecutor<
  Data,
  statement::For<ArgumentId, RAJA::hip_thread_masked_direct<Mask>,
                 EnclosedStmts ...>,
  Types> {

  using stmt_list_t = StatementList<EnclosedStmts ...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
          HipStatementListExecutor<Data, stmt_list_t, NewTypes>;

  using mask_t = Mask;

  using diff_t = segment_diff_type<ArgumentId, Data>;

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    diff_t len = segment_length<ArgumentId>(data);

    diff_t i = mask_t::maskValue((diff_t)threadIdx.x);

    // assign thread id directly to offset
    data.template assign_offset<ArgumentId>(i);

    // execute enclosed statements if in bounds
    enclosed_stmts_t::exec(data, thread_active && (i<len));
  }


  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    // Get enclosed statements
    LaunchDims dims;

    // we need to allocate enough threads for the segment size, and the
    // shifted off bits
    diff_t len = mask_t::max_input_size;

    // request one thread per element in the segment
    set_hip_dim<0>(dims.threads, len);

    // since we are direct-mapping, we REQUIRE len
    set_hip_dim<0>(dims.min_threads, len);

    LaunchDims enclosed_dims = enclosed_stmts_t::calculateDimensions(data);
    return(dims.max(enclosed_dims));
  }
};




/*
 * Executor for thread work sharing loop inside HipKernel.
 * Mapping directly from a warp lane
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename Mask,
          typename ... EnclosedStmts,
          typename Types>
struct HipStatementExecutor<
  Data,
  statement::For<ArgumentId, RAJA::hip_thread_masked_loop<Mask>,
                 EnclosedStmts ...>,
  Types> {

  using stmt_list_t = StatementList<EnclosedStmts ...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
          HipStatementListExecutor<Data, stmt_list_t, NewTypes>;

  using mask_t = Mask;

  using diff_t = segment_diff_type<ArgumentId, Data>;


  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    // masked size strided loop
    diff_t len = segment_length<ArgumentId>(data);
    diff_t i_init = mask_t::maskValue((diff_t)threadIdx.x);
    diff_t i_stride = (diff_t) mask_t::max_masked_size;

    // Iterate through grid stride of chunks
    for (diff_t ii = 0; ii < len; ii += i_stride) {
      diff_t i = ii + i_init;

      // execute enclosed statements if any thread will
      // but mask off threads without work
      bool have_work = i < len;

      // Assign the x thread to the argument
      data.template assign_offset<ArgumentId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active && have_work);
    }
  }


  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    // Get enclosed statements
    LaunchDims dims;

    // we need to allocate enough threads for the segment size, and the
    // shifted off bits
    diff_t len = mask_t::max_input_size;

    // request one thread per element in the segment
    set_hip_dim<0>(dims.threads, len);

    // since we are direct-mapping, we REQUIRE len
    set_hip_dim<0>(dims.min_threads, len);

    LaunchDims enclosed_dims = enclosed_stmts_t::calculateDimensions(data);
    return(dims.max(enclosed_dims));
  }
};


/*
 * Executor for work sharing inside HipKernel.
 * Mapping directly from Indexer to indices
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename Indexer,
          typename... EnclosedStmts,
          typename Types>
struct HipStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::internal::HipIndexDirect<Indexer>, EnclosedStmts...>,
    Types> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      HipStatementListExecutor<Data, stmt_list_t, NewTypes>;

  using diff_t = segment_diff_type<ArgumentId, Data>;


  static inline RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    diff_t len = segment_length<ArgumentId>(data);
    diff_t i = Indexer::template index<diff_t>();

    // Assign the index to the argument
    data.template assign_offset<ArgumentId>(i);

    // execute enclosed statements
    enclosed_stmts_t::exec(data, thread_active && (i < len));
  }

  static inline
  LaunchDims calculateDimensions(Data const &data)
  {
    diff_t len = segment_length<ArgumentId>(data);

    LaunchDims dims = HipIndexDimensioner<Indexer>::get_dimensions(len);

    // since we are direct-mapping, we REQUIRE the given dimensions
    dims.min_threads = dims.threads;
    dims.min_blocks = dims.blocks;

    // combine with enclosed statements
    LaunchDims enclosed_dims = enclosed_stmts_t::calculateDimensions(data);
    return dims.max(enclosed_dims);
  }
};

/*
 * Executor for work sharing inside HipKernel.
 * Provides a strided loop for Indexer.
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename Indexer,
          typename... EnclosedStmts,
          typename Types>
struct HipStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::internal::HipIndexLoop<Indexer>, EnclosedStmts...>,
    Types> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      HipStatementListExecutor<Data, stmt_list_t, NewTypes>;

  using diff_t = segment_diff_type<ArgumentId, Data>;


  static inline RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    // grid stride loop
    diff_t len = segment_length<ArgumentId>(data);
    diff_t i_init = Indexer::template index<diff_t>();
    diff_t i_stride = Indexer::template size<diff_t>();

    // Iterate through chunks
    for (diff_t ii = 0; ii < len; ii += i_stride) {
      diff_t i = ii + i_init;

      // execute enclosed statements if any thread will
      // but mask off threads without work
      bool have_work = i < len;

      // Assign the index to the argument
      data.template assign_offset<ArgumentId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active && have_work);
    }
  }

  static inline
  LaunchDims calculateDimensions(Data const &data)
  {
    diff_t len = segment_length<ArgumentId>(data);

    LaunchDims dims = HipIndexDimensioner<Indexer>::get_dimensions(len);

    // combine with enclosed statements
    LaunchDims enclosed_dims = enclosed_stmts_t::calculateDimensions(data);
    return dims.max(enclosed_dims);
  }
};


/*
 * Executor for sequential loops inside of a HipKernel.
 *
 * This is specialized since it need to execute the loop immediately.
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename... EnclosedStmts,
          typename Types>
struct HipStatementExecutor<
    Data,
    statement::For<ArgumentId, seq_exec, EnclosedStmts...>,
    Types> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
      HipStatementListExecutor<Data, stmt_list_t, NewTypes>;

  using diff_t = segment_diff_type<ArgumentId, Data>;

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    diff_t len = segment_length<ArgumentId>(data);

    for(diff_t i = 0;i < len;++ i){
      // Assign i to the argument
      data.template assign_offset<ArgumentId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active);
    }
  }


  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    return enclosed_stmts_t::calculateDimensions(data);
  }
};

template <typename Data,
          camp::idx_t ArgumentId,
          typename... EnclosedStmts,
          typename Types>
struct HipStatementExecutor<
    Data,
    statement::For<ArgumentId, loop_exec, EnclosedStmts...>,
    Types>
:  HipStatementExecutor<Data, statement::For<ArgumentId, seq_exec, EnclosedStmts...>, Types>
{

};


}  // namespace internal
}  // end namespace RAJA


#endif /* RAJA_policy_hip_kernel_For_HPP */
