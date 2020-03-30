/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for CUDA statement executors.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#ifndef RAJA_policy_cuda_kernel_ForICount_HPP
#define RAJA_policy_cuda_kernel_ForICount_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/cuda/kernel/internal.hpp"


namespace RAJA
{

namespace internal
{



/*
 * Executor for thread work sharing loop inside CudaKernel.
 * Mapping directly from threadIdx.xyz to indices
 * Assigns the loop iterate to offset ArgumentId
 * Assigns the loop count to param ParamId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename ParamId,
          int ThreadDim,
          typename... EnclosedStmts,
          typename Types>
struct CudaStatementExecutor<
    Data,
    statement::ForICount<ArgumentId, ParamId, RAJA::cuda_thread_xyz_direct<ThreadDim>, EnclosedStmts...>,
    Types>
    : public CudaStatementExecutor<
        Data,
        statement::For<ArgumentId, RAJA::cuda_thread_xyz_direct<ThreadDim>, EnclosedStmts...>, Types> {

  using Base = CudaStatementExecutor<
        Data,
        statement::For<ArgumentId, RAJA::cuda_thread_xyz_direct<ThreadDim>, EnclosedStmts...>,
        Types>;

  using typename Base::enclosed_stmts_t;

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    auto len = segment_length<ArgumentId>(data);
    auto i = get_cuda_dim<ThreadDim>(threadIdx);

    // assign thread id directly to offset
    data.template assign_offset<ArgumentId>(i);
    data.template assign_param<ParamId>(i);

    // execute enclosed statements if in bounds
    enclosed_stmts_t::exec(data, thread_active && (i<len));

  }
};





/*
 * Executor for thread work sharing loop inside CudaKernel.
 * Mapping directly from threadIdx.xyz to indices
 * Assigns the loop iterate to offset ArgumentId
 * Assigns the loop count to param ParamId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename ParamId,
          typename ... EnclosedStmts,
          typename Types>
struct CudaStatementExecutor<
  Data,
  statement::ForICount<ArgumentId, ParamId, RAJA::cuda_warp_direct,
                       EnclosedStmts ...>,
  Types>
  : public CudaStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::cuda_warp_direct,
                   EnclosedStmts ...>, Types > {

  using Base = CudaStatementExecutor<
          Data,
          statement::For<ArgumentId, RAJA::cuda_warp_direct,
                         EnclosedStmts ...>, Types >;

  using typename Base::enclosed_stmts_t;

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    auto len = segment_length<ArgumentId>(data);
    auto i = get_cuda_dim<0>(threadIdx);

    // assign thread id directly to offset
    data.template assign_offset<ArgumentId>(i);
    data.template assign_param<ParamId>(i);

    // execute enclosed statements if in bounds
    enclosed_stmts_t::exec(data, thread_active && (i<len));

  }
};


/*
 * Executor for thread work sharing loop inside CudaKernel.
 * Mapping directly from threadIdx.xyz to indices
 * Assigns the loop iterate to offset ArgumentId
 * Assigns the loop count to param ParamId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename ParamId,
          typename ... EnclosedStmts,
          typename Types>
struct CudaStatementExecutor<
  Data,
  statement::ForICount<ArgumentId, ParamId, RAJA::cuda_warp_loop,
                       EnclosedStmts ...>, Types >
  : public CudaStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::cuda_warp_loop,
                   EnclosedStmts ...>, Types > {

  using Base = CudaStatementExecutor<
          Data,
          statement::For<ArgumentId, RAJA::cuda_warp_loop,
                         EnclosedStmts ...>, Types >;

  using typename Base::enclosed_stmts_t;

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    // block stride loop
    int len = segment_length<ArgumentId>(data);
    //auto i0 = threadIdx.x;
    //auto i_stride = RAJA::policy::cuda::WARP_SIZE;
    //auto i = i0;
    auto &i = camp::get<ArgumentId>(data.offset_tuple);
    i = threadIdx.x;
    for( ; i < len; i += RAJA::policy::cuda::WARP_SIZE){

      // Assign the x thread to the argument
      //  data.template assign_offset<ArgumentId>(i);
      data.template assign_param<ParamId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active);
    }
    // do we need one more masked iteration?
    if(i - threadIdx.x < len){
      // execute enclosed statements one more time, but masking them off
      // this is because there's at least one thread that isn't masked off
      // that is still executing the above loop
      enclosed_stmts_t::exec(data, false);
    }
  }
};


/*
 * Executor for thread work sharing loop inside CudaKernel.
 * Mapping directly from a warp lane
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename ParamId,
          typename Mask,
          typename ... EnclosedStmts,
          typename Types>
struct CudaStatementExecutor<
  Data,
  statement::ForICount<ArgumentId, ParamId,
                       RAJA::cuda_warp_masked_direct<Mask>,
                       EnclosedStmts ...>, Types >
  : public CudaStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::cuda_warp_masked_direct<Mask>,
                   EnclosedStmts ...>, Types > {

  using Base = CudaStatementExecutor<
          Data,
          statement::For<ArgumentId, RAJA::cuda_warp_masked_direct<Mask>,
                         EnclosedStmts ...>, Types >;

  using stmt_list_t = StatementList<EnclosedStmts ...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
          CudaStatementListExecutor<Data, stmt_list_t, NewTypes>;

  using mask_t = Mask;

  static_assert(mask_t::max_masked_size <= RAJA::policy::cuda::WARP_SIZE,
                "BitMask is too large for CUDA warp size");

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    auto len = segment_length<ArgumentId>(data);

    auto i = mask_t::maskValue(threadIdx.x);

    // assign thread id directly to offset
    data.template assign_offset<ArgumentId>(i);
    data.template assign_param<ParamId>(i);

    // execute enclosed statements if in bounds
    enclosed_stmts_t::exec(data, thread_active && (i<len));
  }

};



/*
 * Executor for thread work sharing loop inside CudaKernel.
 * Mapping directly from a warp lane
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename ParamId,
          typename Mask,
          typename ... EnclosedStmts,
          typename Types>
struct CudaStatementExecutor<
  Data,
  statement::ForICount<ArgumentId, ParamId,
                       RAJA::cuda_warp_masked_loop<Mask>,
                       EnclosedStmts ...>, Types >
  : public CudaStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::cuda_warp_masked_loop<Mask>,
                   EnclosedStmts ...>, Types > {

  using Base = CudaStatementExecutor<
          Data,
          statement::For<ArgumentId, RAJA::cuda_warp_masked_loop<Mask>,
                         EnclosedStmts ...>, Types >;

  using stmt_list_t = StatementList<EnclosedStmts ...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
          CudaStatementListExecutor<Data, stmt_list_t, NewTypes>;

  using mask_t = Mask;

  static_assert(mask_t::max_masked_size <= RAJA::policy::cuda::WARP_SIZE,
                "BitMask is too large for CUDA warp size");

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    // masked size strided loop
    int len = segment_length<ArgumentId>(data);
    auto i = mask_t::maskValue(threadIdx.x);
    for( ; i < len; i += (int) mask_t::max_masked_size){

      // Assign the x thread to the argument and param
      data.template assign_offset<ArgumentId>(i);
      data.template assign_param<ParamId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active);
    }
    // do we need one more masked iteration?
    if(i - mask_t::maskValue(threadIdx.x) < len){
      // execute enclosed statements one more time, but masking them off
      // this is because there's at least one thread that isn't masked off
      // that is still executing the above loop
      enclosed_stmts_t::exec(data, false);
    }
  }

};




/*
 * Executor for thread work sharing loop inside CudaKernel.
 * Mapping directly from a warp lane
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename ParamId,
          typename Mask,
          typename ... EnclosedStmts,
          typename Types>
struct CudaStatementExecutor<
  Data,
  statement::ForICount<ArgumentId, ParamId,
                       RAJA::cuda_thread_masked_direct<Mask>,
                       EnclosedStmts ...>, Types >
  : public CudaStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::cuda_thread_masked_direct<Mask>,
                   EnclosedStmts ...>, Types > {

  using Base = CudaStatementExecutor<
          Data,
          statement::For<ArgumentId, RAJA::cuda_thread_masked_direct<Mask>,
                         EnclosedStmts ...>, Types >;

  using stmt_list_t = StatementList<EnclosedStmts ...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
          CudaStatementListExecutor<Data, stmt_list_t, NewTypes>;

  using mask_t = Mask;

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    auto len = segment_length<ArgumentId>(data);

    auto i = mask_t::maskValue(threadIdx.x);

    // assign thread id directly to offset
    data.template assign_offset<ArgumentId>(i);
    data.template assign_param<ParamId>(i);

    // execute enclosed statements if in bounds
    enclosed_stmts_t::exec(data, thread_active && (i<len));
  }

};





/*
 * Executor for thread work sharing loop inside CudaKernel.
 * Mapping directly from a warp lane
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename ParamId,
          typename Mask,
          typename ... EnclosedStmts,
          typename Types>
struct CudaStatementExecutor<
  Data,
  statement::ForICount<ArgumentId, ParamId,
                       RAJA::cuda_thread_masked_loop<Mask>,
                       EnclosedStmts ...>, Types >
  : public CudaStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::cuda_thread_masked_loop<Mask>,
                   EnclosedStmts ...>, Types > {

  using Base = CudaStatementExecutor<
          Data,
          statement::For<ArgumentId, RAJA::cuda_thread_masked_loop<Mask>,
                         EnclosedStmts ...>, Types >;

  using stmt_list_t = StatementList<EnclosedStmts ...>;

  // Set the argument type for this loop
  using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

  using enclosed_stmts_t =
          CudaStatementListExecutor<Data, stmt_list_t, NewTypes>;

  using mask_t = Mask;

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    // masked size strided loop
    int len = segment_length<ArgumentId>(data);
    int i = mask_t::maskValue(threadIdx.x);
    for( ; i < len; i += (int) mask_t::max_masked_size){

      // Assign the x thread to the argument
      data.template assign_offset<ArgumentId>(i);
      data.template assign_param<ParamId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active);
    }
    // do we need one more masked iteration?
    if(i - mask_t::maskValue(threadIdx.x) < len){
      // execute enclosed statements one more time, but masking them off
      // this is because there's at least one thread that isn't masked off
      // that is still executing the above loop
      enclosed_stmts_t::exec(data, false);
    }
  }

};





/*
 * Executor for thread work sharing loop inside CudaKernel.
 * Provides a block-stride loop (stride of blockDim.xyz) for
 * each thread in xyz.
 * Assigns the loop iterate to offset ArgumentId
 * Assigns the loop offset to param ParamId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename ParamId,
          int ThreadDim,
          int MinThreads,
          typename... EnclosedStmts,
          typename Types>
struct CudaStatementExecutor<
    Data,
    statement::ForICount<ArgumentId, ParamId, RAJA::cuda_thread_xyz_loop<ThreadDim, MinThreads>, EnclosedStmts...>,
    Types>
    : public CudaStatementExecutor<
        Data,
        statement::For<ArgumentId, RAJA::cuda_thread_xyz_loop<ThreadDim, MinThreads>, EnclosedStmts...>,
        Types> {

  using Base = CudaStatementExecutor<
        Data,
        statement::For<ArgumentId, RAJA::cuda_thread_xyz_loop<ThreadDim, MinThreads>, EnclosedStmts...>,
        Types>;

  using typename Base::enclosed_stmts_t;

  static
  inline RAJA_DEVICE void exec(Data &data, bool thread_active)
  {
    // block stride loop
    auto len = segment_length<ArgumentId>(data);
    auto i0 = get_cuda_dim<ThreadDim>(threadIdx);
    auto i_stride = get_cuda_dim<ThreadDim>(blockDim);
    auto i = i0;
    for(;i < len;i += i_stride){

      // Assign the x thread to the argument
      data.template assign_offset<ArgumentId>(i);
      data.template assign_param<ParamId>(i);

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
};



/*
 * Executor for block work sharing inside CudaKernel.
 * Provides a direct mapping for each block in xyz.
 * Assigns the loop index to offset ArgumentId
 * Assigns the loop index to param ParamId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename ParamId,
          int BlockDim,
          typename... EnclosedStmts,
          typename Types>
struct CudaStatementExecutor<
    Data,
    statement::ForICount<ArgumentId, ParamId, RAJA::cuda_block_xyz_direct<BlockDim>, EnclosedStmts...>,
    Types>
    : public CudaStatementExecutor<
        Data,
        statement::For<ArgumentId, RAJA::cuda_block_xyz_direct<BlockDim>, EnclosedStmts...>,
        Types> {

  using Base = CudaStatementExecutor<
      Data,
      statement::For<ArgumentId, RAJA::cuda_block_xyz_direct<BlockDim>, EnclosedStmts...>,
      Types>;

  using typename Base::enclosed_stmts_t;

  static
  inline RAJA_DEVICE void exec(Data &data, bool thread_active)
  {
    // grid stride loop
    auto len = segment_length<ArgumentId>(data);
    auto i = get_cuda_dim<BlockDim>(blockIdx);

    if (i < len) {

      // Assign the x thread to the argument
      data.template assign_offset<ArgumentId>(i);
      data.template assign_param<ParamId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active);
    }
  }
};

/*
 * Executor for block work sharing inside CudaKernel.
 * Provides a grid-stride loop (stride of gridDim.xyz) for
 * each block in xyz.
 * Assigns the loop index to offset ArgumentId
 * Assigns the loop index to param ParamId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename ParamId,
          int BlockDim,
          typename... EnclosedStmts,
          typename Types>
struct CudaStatementExecutor<
    Data,
    statement::ForICount<ArgumentId, ParamId, RAJA::cuda_block_xyz_loop<BlockDim>, EnclosedStmts...>,
    Types>
    : public CudaStatementExecutor<
        Data,
        statement::For<ArgumentId, RAJA::cuda_block_xyz_loop<BlockDim>, EnclosedStmts...>,
        Types> {

  using Base = CudaStatementExecutor<
      Data,
      statement::For<ArgumentId, RAJA::cuda_block_xyz_loop<BlockDim>, EnclosedStmts...>,
      Types>;

  using typename Base::enclosed_stmts_t;

  static
  inline RAJA_DEVICE void exec(Data &data, bool thread_active)
  {
    // grid stride loop
    auto len = segment_length<ArgumentId>(data);
    auto i0 = get_cuda_dim<BlockDim>(blockIdx);
    auto i_stride = get_cuda_dim<BlockDim>(gridDim);
    for(auto i = i0;i < len;i += i_stride){

      // Assign the x thread to the argument
      data.template assign_offset<ArgumentId>(i);
      data.template assign_param<ParamId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active);
    }
  }
};


/*
 * Executor for sequential loops inside of a CudaKernel.
 *
 * This is specialized since it need to execute the loop immediately.
 * Assigns the loop index to offset ArgumentId
 * Assigns the loop index to param ParamId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename ParamId,
          typename... EnclosedStmts,
          typename Types>
struct CudaStatementExecutor<
    Data,
    statement::ForICount<ArgumentId, ParamId, seq_exec, EnclosedStmts...>, Types >
    : public CudaStatementExecutor<
        Data,
        statement::For<ArgumentId, seq_exec, EnclosedStmts...>, Types > {

  using Base = CudaStatementExecutor<
      Data,
      statement::For<ArgumentId, seq_exec, EnclosedStmts...>, Types >;

  using typename Base::enclosed_stmts_t;

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
      data.template assign_param<ParamId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active);
    }
  }
};





}  // namespace internal
}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
