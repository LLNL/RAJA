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


#ifndef RAJA_policy_hip_kernel_ForICount_HPP
#define RAJA_policy_hip_kernel_ForICount_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/hip/kernel/internal.hpp"


namespace RAJA
{

namespace internal
{



/*
 * Executor for thread work sharing loop inside HipKernel.
 * Mapping directly from threadIdx.xyz to indices
 * Assigns the loop iterate to offset ArgumentId
 * Assigns the loop count to param ParamId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename ParamId,
          int ThreadDim,
          typename... EnclosedStmts>
struct HipStatementExecutor<
    Data,
    statement::ForICount<ArgumentId, ParamId, RAJA::hip_thread_xyz_direct<ThreadDim>, EnclosedStmts...>>
    : public HipStatementExecutor<
        Data,
        statement::For<ArgumentId, RAJA::hip_thread_xyz_direct<ThreadDim>, EnclosedStmts...>> {

  using Base = HipStatementExecutor<
        Data,
        statement::For<ArgumentId, RAJA::hip_thread_xyz_direct<ThreadDim>, EnclosedStmts...>>;

  using typename Base::enclosed_stmts_t;

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    auto len = segment_length<ArgumentId>(data);
    auto i = get_hip_dim<ThreadDim>(dim3(threadIdx.x,threadIdx.y,threadIdx.z));

    // assign thread id directly to offset
    data.template assign_offset<ArgumentId>(i);
    data.template assign_param<ParamId>(i);

    // execute enclosed statements if in bounds
    enclosed_stmts_t::exec(data, thread_active && (i<len));

  }
};





/*
 * Executor for thread work sharing loop inside HipKernel.
 * Mapping directly from threadIdx.xyz to indices
 * Assigns the loop iterate to offset ArgumentId
 * Assigns the loop count to param ParamId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename ParamId,
          typename ... EnclosedStmts>
struct HipStatementExecutor<
  Data,
  statement::ForICount<ArgumentId, ParamId, RAJA::hip_warp_direct,
                       EnclosedStmts ...> >
  : public HipStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::hip_warp_direct,
                   EnclosedStmts ...> > {

  using Base = HipStatementExecutor<
          Data,
          statement::For<ArgumentId, RAJA::hip_warp_direct,
                         EnclosedStmts ...> >;

  using typename Base::enclosed_stmts_t;

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    auto len = segment_length<ArgumentId>(data);
    auto i = get_hip_dim<0>(dim3(threadIdx.x,threadIdx.y,threadIdx.z));

    // assign thread id directly to offset
    data.template assign_offset<ArgumentId>(i);
    data.template assign_param<ParamId>(i);

    // execute enclosed statements if in bounds
    enclosed_stmts_t::exec(data, thread_active && (i<len));

  }
};


/*
 * Executor for thread work sharing loop inside HipKernel.
 * Mapping directly from threadIdx.xyz to indices
 * Assigns the loop iterate to offset ArgumentId
 * Assigns the loop count to param ParamId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename ParamId,
          typename ... EnclosedStmts>
struct HipStatementExecutor<
  Data,
  statement::ForICount<ArgumentId, ParamId, RAJA::hip_warp_loop,
                       EnclosedStmts ...> >
  : public HipStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::hip_warp_loop,
                   EnclosedStmts ...> > {

  using Base = HipStatementExecutor<
          Data,
          statement::For<ArgumentId, RAJA::hip_warp_loop,
                         EnclosedStmts ...> >;

  using typename Base::enclosed_stmts_t;

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    // block stride loop
    int len = segment_length<ArgumentId>(data);
    //auto i0 = threadIdx.x;
    //auto i_stride = RAJA::policy::hip::WARP_SIZE;
    //auto i = i0;
    auto &i = camp::get<ArgumentId>(data.offset_tuple);
    i = threadIdx.x;
    for( ; i < len; i += RAJA::policy::hip::WARP_SIZE){

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
 * Executor for thread work sharing loop inside HipKernel.
 * Mapping directly from a warp lane
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename ParamId,
          typename Mask,
          typename ... EnclosedStmts>
struct HipStatementExecutor<
  Data,
  statement::ForICount<ArgumentId, ParamId,
                       RAJA::hip_warp_masked_direct<Mask>,
                       EnclosedStmts ...> >
  : public HipStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::hip_warp_masked_direct<Mask>,
                   EnclosedStmts ...> > {

  using Base = HipStatementExecutor<
          Data,
          statement::For<ArgumentId, RAJA::hip_warp_masked_direct<Mask>,
                         EnclosedStmts ...> >;

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
    data.template assign_param<ParamId>(i);

    // execute enclosed statements if in bounds
    enclosed_stmts_t::exec(data, thread_active && (i<len));
  }

};



/*
 * Executor for thread work sharing loop inside HipKernel.
 * Mapping directly from a warp lane
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename ParamId,
          typename Mask,
          typename ... EnclosedStmts>
struct HipStatementExecutor<
  Data,
  statement::ForICount<ArgumentId, ParamId,
                       RAJA::hip_warp_masked_loop<Mask>,
                       EnclosedStmts ...> >
  : public HipStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::hip_warp_masked_loop<Mask>,
                   EnclosedStmts ...> > {

  using Base = HipStatementExecutor<
          Data,
          statement::For<ArgumentId, RAJA::hip_warp_masked_loop<Mask>,
                         EnclosedStmts ...> >;

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
    auto i = mask_t::maskValue((int)threadIdx.x);
    for( ; i < len; i += (int) mask_t::max_masked_size){

      // Assign the x thread to the argument and param
      data.template assign_offset<ArgumentId>(i);
      data.template assign_param<ParamId>(i);

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

};




/*
 * Executor for thread work sharing loop inside HipKernel.
 * Mapping directly from a warp lane
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename ParamId,
          typename Mask,
          typename ... EnclosedStmts>
struct HipStatementExecutor<
  Data,
  statement::ForICount<ArgumentId, ParamId,
                       RAJA::hip_thread_masked_direct<Mask>,
                       EnclosedStmts ...> >
  : public HipStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::hip_thread_masked_direct<Mask>,
                   EnclosedStmts ...> > {

  using Base = HipStatementExecutor<
          Data,
          statement::For<ArgumentId, RAJA::hip_thread_masked_direct<Mask>,
                         EnclosedStmts ...> >;

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
    data.template assign_param<ParamId>(i);

    // execute enclosed statements if in bounds
    enclosed_stmts_t::exec(data, thread_active && (i<len));
  }

};





/*
 * Executor for thread work sharing loop inside HipKernel.
 * Mapping directly from a warp lane
 * Assigns the loop index to offset ArgumentId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename ParamId,
          typename Mask,
          typename ... EnclosedStmts>
struct HipStatementExecutor<
  Data,
  statement::ForICount<ArgumentId, ParamId,
                       RAJA::hip_thread_masked_loop<Mask>,
                       EnclosedStmts ...> >
  : public HipStatementExecutor<
    Data,
    statement::For<ArgumentId, RAJA::hip_thread_masked_loop<Mask>,
                   EnclosedStmts ...> > {

  using Base = HipStatementExecutor<
          Data,
          statement::For<ArgumentId, RAJA::hip_thread_masked_loop<Mask>,
                         EnclosedStmts ...> >;

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
      data.template assign_param<ParamId>(i);

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

};





/*
 * Executor for thread work sharing loop inside HipKernel.
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
          typename... EnclosedStmts>
struct HipStatementExecutor<
    Data,
    statement::ForICount<ArgumentId, ParamId, RAJA::hip_thread_xyz_loop<ThreadDim, MinThreads>, EnclosedStmts...>>
    : public HipStatementExecutor<
        Data,
        statement::For<ArgumentId, RAJA::hip_thread_xyz_loop<ThreadDim, MinThreads>, EnclosedStmts...>> {

  using Base = HipStatementExecutor<
        Data,
        statement::For<ArgumentId, RAJA::hip_thread_xyz_loop<ThreadDim, MinThreads>, EnclosedStmts...>>;

  using typename Base::enclosed_stmts_t;

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
 * Executor for block work sharing inside HipKernel.
 * Provides a direct mapping of each block in xyz.
 * Assigns the loop index to offset ArgumentId
 * Assigns the loop index to param ParamId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename ParamId,
          int BlockDim,
          typename... EnclosedStmts>
struct HipStatementExecutor<
    Data,
    statement::ForICount<ArgumentId, ParamId, RAJA::hip_block_xyz_direct<BlockDim>, EnclosedStmts...>>
    : public HipStatementExecutor<
        Data,
        statement::For<ArgumentId, RAJA::hip_block_xyz_direct<BlockDim>, EnclosedStmts...>> {

  using Base = HipStatementExecutor<
      Data,
      statement::For<ArgumentId, RAJA::hip_block_xyz_direct<BlockDim>, EnclosedStmts...>>;

  using typename Base::enclosed_stmts_t;

  static
  inline RAJA_DEVICE void exec(Data &data, bool thread_active)
  {
    // grid stride loop
    auto len = segment_length<ArgumentId>(data);
    auto i = get_hip_dim<BlockDim>(blockIdx);

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
 * Executor for block work sharing inside HipKernel.
 * Provides a grid-stride loop (stride of gridDim.xyz) for
 * each block in xyz.
 * Assigns the loop index to offset ArgumentId
 * Assigns the loop index to param ParamId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename ParamId,
          int BlockDim,
          typename... EnclosedStmts>
struct HipStatementExecutor<
    Data,
    statement::ForICount<ArgumentId, ParamId, RAJA::hip_block_xyz_loop<BlockDim>, EnclosedStmts...>>
    : public HipStatementExecutor<
        Data,
        statement::For<ArgumentId, RAJA::hip_block_xyz_loop<BlockDim>, EnclosedStmts...>> {

  using Base = HipStatementExecutor<
      Data,
      statement::For<ArgumentId, RAJA::hip_block_xyz_loop<BlockDim>, EnclosedStmts...>>;

  using typename Base::enclosed_stmts_t;

  static
  inline RAJA_DEVICE void exec(Data &data, bool thread_active)
  {
    // grid stride loop
    auto len = segment_length<ArgumentId>(data);
    auto i0 = get_hip_dim<BlockDim>(blockIdx);
    auto i_stride = get_hip_dim<BlockDim>(gridDim);
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
 * Executor for sequential loops inside of a HipKernel.
 *
 * This is specialized since it need to execute the loop immediately.
 * Assigns the loop index to offset ArgumentId
 * Assigns the loop index to param ParamId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename ParamId,
          typename... EnclosedStmts>
struct HipStatementExecutor<
    Data,
    statement::ForICount<ArgumentId, ParamId, seq_exec, EnclosedStmts...> >
    : public HipStatementExecutor<
        Data,
        statement::For<ArgumentId, seq_exec, EnclosedStmts...> > {

  using Base = HipStatementExecutor<
      Data,
      statement::For<ArgumentId, seq_exec, EnclosedStmts...> >;

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
