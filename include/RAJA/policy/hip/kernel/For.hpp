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
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
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
 * Mapping directly from threadIdx.xyz to indices
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          int ThreadDim,
          typename... EnclosedStmts>
struct HipStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::hip_thread_xyz_direct<ThreadDim>, EnclosedStmts...>> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  using enclosed_stmts_t =
      HipStatementListExecutor<Data, stmt_list_t>;


  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    auto len = segment_length<ArgumentId>(data);
    auto i = get_hip_dim<ThreadDim>(dim3(threadIdx.x,threadIdx.y,threadIdx.z));

    // assign thread id directly to offset
    data.template assign_offset<ArgumentId>(i);

    // execute enclosed statements if in bounds
    enclosed_stmts_t::exec(data, thread_active && (i<len));
  }


  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    auto len = segment_length<ArgumentId>(data);

    // request one thread per element in the segment
    LaunchDims dims;
    set_hip_dim<ThreadDim>(dims.threads, len);

    // since we are direct-mapping, we REQUIRE len
    set_hip_dim<ThreadDim>(dims.min_threads, len);

    // combine with enclosed statements
    LaunchDims enclosed_dims = enclosed_stmts_t::calculateDimensions(data);
    return dims.max(enclosed_dims);
  }
};


/*
 * Executor for thread work sharing loop inside HipKernel.
 * Mapping directly from a warp lane
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename... EnclosedStmts>
struct HipStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::hip_warp_direct, EnclosedStmts...>> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  using enclosed_stmts_t =
      HipStatementListExecutor<Data, stmt_list_t>;


  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    auto len = segment_length<ArgumentId>(data);
    auto i = threadIdx.x;

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
    int len = RAJA::policy::hip::WARP_SIZE;

    // request one thread per element in the segment
    set_hip_dim<0>(dims.threads, len);

    // since we are direct-mapping, we REQUIRE len
    set_hip_dim<0>(dims.min_threads, len);

    return dims;
  }
};


/*
 * Executor for thread work sharing loop inside HipKernel.
 * Provides a block-stride loop (stride of blockDim.xyz) for
 * each thread in xyz.
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          int ThreadDim,
          int MinThreads,
          typename... EnclosedStmts>
struct HipStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::hip_thread_xyz_loop<ThreadDim, MinThreads>, EnclosedStmts...>> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  using enclosed_stmts_t =
      HipStatementListExecutor<Data, stmt_list_t>;


  static
  inline RAJA_DEVICE void exec(Data &data, bool thread_active)
  {
    // block stride loop
    auto len = segment_length<ArgumentId>(data);
    auto i0 = get_hip_dim<ThreadDim>(dim3(threadIdx.x,threadIdx.y,threadIdx.z));
    auto i_stride = get_hip_dim<ThreadDim>(dim3(blockDim.x,blockDim.y,blockDim.z));
    auto i = i0;
    for(;i < len;i += i_stride){

      // Assign the x thread to the argument
      data.template assign_offset<ArgumentId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active);
    }
    // do we need one more masked iteration?
    if(i - i0 < len)
    {
      // execute enclosed statements one more time, but masking them off
      // this is because there's at least one thread that isn't masked off
      // that is still executing the above loop
      enclosed_stmts_t::exec(data, false);
    }
  }


  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    auto len = segment_length<ArgumentId>(data);

    // request one thread per element in the segment
    LaunchDims dims;
    set_hip_dim<ThreadDim>(dims.threads, len);

    // but, since we are looping, we only need 1 thread, or whatever
    // the user specified for MinThreads
    set_hip_dim<ThreadDim>(dims.min_threads, MinThreads);

    // combine with enclosed statements
    LaunchDims enclosed_dims = enclosed_stmts_t::calculateDimensions(data);
    return dims.max(enclosed_dims);
  }
};


/*
 * Executor for thread work sharing loop inside HipKernel.
 * Provides a warp-stride loop for each thread inside of a warp.
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename... EnclosedStmts>
struct HipStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::hip_warp_loop, EnclosedStmts...>> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  using enclosed_stmts_t =
      HipStatementListExecutor<Data, stmt_list_t>;


  static
  inline RAJA_DEVICE void exec(Data &data, bool thread_active)
  {
    // block stride loop
    int len = segment_length<ArgumentId>(data);
    int i0 = threadIdx.x;
    auto i_stride = RAJA::policy::hip::WARP_SIZE;
    int i = i0;
    for(;i < len;i += i_stride){

      // Assign the x thread to the argument
      data.template assign_offset<ArgumentId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active);
    }
    // do we need one more masked iteration?
    if(i - i0 < len)
    {
      // execute enclosed statements one more time, but masking them off
      // this is because there's at least one thread that isn't masked off
      // that is still executing the above loop
      enclosed_stmts_t::exec(data, false);
    }
  }


  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    // Get enclosed statements
    LaunchDims dims = enclosed_stmts_t::calculateDimensions(data);

    // we always get EXACTLY one warp by allocating one warp in the X dimension
    int len = RAJA::policy::hip::WARP_SIZE;

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
          typename ... EnclosedStmts>
struct HipStatementExecutor<
  Data,
  statement::For<ArgumentId, RAJA::hip_warp_masked_direct<Mask>,
                 EnclosedStmts ...> > {

  using stmt_list_t = StatementList<EnclosedStmts ...>;

  using enclosed_stmts_t =
          HipStatementListExecutor<Data, stmt_list_t>;

  using mask_t = Mask;

  static_assert(mask_t::max_masked_size <= RAJA::policy::hip::WARP_SIZE,
                "BitMask is too large for HIP warp size");

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    auto len = segment_length<ArgumentId>(data);

    auto i = mask_t::maskValue((int)threadIdx.x);

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
    int len = RAJA::policy::hip::WARP_SIZE;

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
          typename ... EnclosedStmts>
struct HipStatementExecutor<
  Data,
  statement::For<ArgumentId, RAJA::hip_warp_masked_loop<Mask>,
                 EnclosedStmts ...> > {

  using stmt_list_t = StatementList<EnclosedStmts ...>;

  using enclosed_stmts_t =
          HipStatementListExecutor<Data, stmt_list_t>;

  using mask_t = Mask;

  static_assert(mask_t::max_masked_size <= RAJA::policy::hip::WARP_SIZE,
                "BitMask is too large for HIP warp size");

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    // masked size strided loop
    int len = segment_length<ArgumentId>(data);
    int i = mask_t::maskValue((int)threadIdx.x);
    for( ; i < len; i += (int) mask_t::max_masked_size){

      // Assign the x thread to the argument
      data.template assign_offset<ArgumentId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active);
    }
    // do we need one more masked iteration?
    if(i - mask_t::maskValue((int)threadIdx.x) < len){
      // execute enclosed statements one more time, but masking them off
      // this is because there's at least one thread that isn't masked off
      // that is still executing the above loop
      enclosed_stmts_t::exec(data, false);
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
    int len = RAJA::policy::hip::WARP_SIZE;

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
          typename ... EnclosedStmts>
struct HipStatementExecutor<
  Data,
  statement::For<ArgumentId, RAJA::hip_thread_masked_direct<Mask>,
                 EnclosedStmts ...> > {

  using stmt_list_t = StatementList<EnclosedStmts ...>;

  using enclosed_stmts_t =
          HipStatementListExecutor<Data, stmt_list_t>;

  using mask_t = Mask;

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    auto len = segment_length<ArgumentId>(data);

    auto i = mask_t::maskValue((int)threadIdx.x);

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
    int len = mask_t::max_input_size;

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
          typename ... EnclosedStmts>
struct HipStatementExecutor<
  Data,
  statement::For<ArgumentId, RAJA::hip_thread_masked_loop<Mask>,
                 EnclosedStmts ...> > {

  using stmt_list_t = StatementList<EnclosedStmts ...>;

  using enclosed_stmts_t =
          HipStatementListExecutor<Data, stmt_list_t>;

  using mask_t = Mask;


  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    // masked size strided loop
    int len = segment_length<ArgumentId>(data);
    int i = mask_t::maskValue((int)threadIdx.x);
    for( ; i < len; i += (int) mask_t::max_masked_size){

      // Assign the x thread to the argument
      data.template assign_offset<ArgumentId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active);
    }
    // do we need one more masked iteration?
    if(i - mask_t::maskValue((int)threadIdx.x) < len){
      // execute enclosed statements one more time, but masking them off
      // this is because there's at least one thread that isn't masked off
      // that is still executing the above loop
      enclosed_stmts_t::exec(data, false);
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
    int len = mask_t::max_input_size;

    // request one thread per element in the segment
    set_hip_dim<0>(dims.threads, len);

    // since we are direct-mapping, we REQUIRE len
    set_hip_dim<0>(dims.min_threads, len);

    LaunchDims enclosed_dims = enclosed_stmts_t::calculateDimensions(data);
    return(dims.max(enclosed_dims));
  }
};


/*
 * Executor for block work sharing inside HipKernel.
 * Provides a grid-stride loop (stride of gridDim.xyz) for
 * each block in xyz.
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          int BlockDim,
          typename... EnclosedStmts>
struct HipStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::hip_block_xyz_direct<BlockDim>, EnclosedStmts...>> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  using enclosed_stmts_t =
      HipStatementListExecutor<Data, stmt_list_t>;


  static
  inline RAJA_DEVICE void exec(Data &data, bool thread_active)
  {
    // grid stride loop
    auto len = segment_length<ArgumentId>(data);
    auto i = get_hip_dim<BlockDim>(dim3(blockIdx.x,blockIdx.y,blockIdx.z));

    if (i < len) {

      // Assign the x thread to the argument
      data.template assign_offset<ArgumentId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active);
    }
  }


  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    auto len = segment_length<ArgumentId>(data);

    // request one block per element in the segment
    LaunchDims dims;
    set_hip_dim<BlockDim>(dims.blocks, len);

    // since we are direct-mapping, we REQUIRE len
    set_hip_dim<BlockDim>(dims.min_blocks, len);

    // combine with enclosed statements
    LaunchDims enclosed_dims = enclosed_stmts_t::calculateDimensions(data);
    return dims.max(enclosed_dims);
  }
};

/*
 * Executor for block work sharing inside HipKernel.
 * Provides a grid-stride loop (stride of gridDim.xyz) for
 * each block in xyz.
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          int BlockDim,
          typename... EnclosedStmts>
struct HipStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::hip_block_xyz_loop<BlockDim>, EnclosedStmts...>> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  using enclosed_stmts_t =
      HipStatementListExecutor<Data, stmt_list_t>;


  static
  inline RAJA_DEVICE void exec(Data &data, bool thread_active)
  {
    // grid stride loop
    auto len = segment_length<ArgumentId>(data);
    auto i0 = get_hip_dim<BlockDim>(dim3(blockIdx.x,blockIdx.y,blockIdx.z));
    auto i_stride = get_hip_dim<BlockDim>(dim3(gridDim.x,gridDim.y,gridDim.z));
    for(auto i = i0;i < len;i += i_stride){

      // Assign the x thread to the argument
      data.template assign_offset<ArgumentId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active);
    }
  }


  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    auto len = segment_length<ArgumentId>(data);

    // request one block per element in the segment
    LaunchDims dims;
    set_hip_dim<BlockDim>(dims.blocks, len);

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
          typename... EnclosedStmts>
struct HipStatementExecutor<
    Data,
    statement::For<ArgumentId, seq_exec, EnclosedStmts...> > {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  using enclosed_stmts_t =
      HipStatementListExecutor<Data, stmt_list_t>;

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {

    using idx_type = camp::decay<decltype(camp::get<ArgumentId>(data.offset_tuple))>;

    idx_type len = segment_length<ArgumentId>(data);

    for(idx_type i = 0;i < len;++ i){
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




}  // namespace internal
}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
