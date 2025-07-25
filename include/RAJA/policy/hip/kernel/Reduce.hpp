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
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#ifndef RAJA_policy_hip_kernel_Reduce_HPP
#define RAJA_policy_hip_kernel_Reduce_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/hip/kernel/internal.hpp"

namespace RAJA
{

namespace internal
{


//
// Executor that handles reductions across a single HIP thread block
//
template<typename Data,
         template<typename...>
         class ReduceOperator,
         typename ParamId,
         typename... EnclosedStmts,
         typename Types>
struct HipStatementExecutor<Data,
                            statement::Reduce<RAJA::hip_block_reduce,
                                              ReduceOperator,
                                              ParamId,
                                              EnclosedStmts...>,
                            Types>
{

  using stmt_list_t = StatementList<EnclosedStmts...>;

  using enclosed_stmts_t = HipStatementListExecutor<Data, stmt_list_t, Types>;

  static inline RAJA_DEVICE void exec(Data& data, bool thread_active)
  {
    // block reduce on the specified parameter
    auto value    = data.template get_param<ParamId>();
    using value_t = decltype(value);
    value_t ident = value_t();

    // if this thread isn't active, just set it to the identity
    if (!thread_active)
    {
      value = ident;
    }

    // Call out existing block reduction algorithm that we use for
    // reduction objects
    using combiner_t =
        RAJA::reduce::detail::op_adapter<value_t, ReduceOperator>;
    value_t new_value = RAJA::hip::impl::block_reduce<combiner_t>(value, ident);


    // execute enclosed statements, and mask off everyone but thread 0
    thread_active = threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0;
    if (thread_active)
    {
      // Only update to new value on root thread
      data.template assign_param<ParamId>(new_value);
    }
    enclosed_stmts_t::exec(data, thread_active);
  }

  static inline LaunchDims calculateDimensions(Data const& data)
  {
    // combine with enclosed statements
    LaunchDims enclosed_dims = enclosed_stmts_t::calculateDimensions(data);
    return enclosed_dims;
  }
};

//
// Executor that handles reductions across a single HIP thread warp
//
template<typename Data,
         template<typename...>
         class ReduceOperator,
         typename ParamId,
         typename... EnclosedStmts,
         typename Types>
struct HipStatementExecutor<Data,
                            statement::Reduce<RAJA::hip_warp_reduce,
                                              ReduceOperator,
                                              ParamId,
                                              EnclosedStmts...>,
                            Types>
{

  using stmt_list_t = StatementList<EnclosedStmts...>;

  using enclosed_stmts_t = HipStatementListExecutor<Data, stmt_list_t, Types>;

  static inline RAJA_DEVICE void exec(Data& data, bool thread_active)
  {
    // block reduce on the specified parameter
    auto value    = data.template get_param<ParamId>();
    using value_t = decltype(value);
    value_t ident = value_t();

    // if this thread isn't active, just set it to the identity
    if (!thread_active)
    {
      value = ident;
    }

    // Call warp reduction routine
    using combiner_t =
        RAJA::reduce::detail::op_adapter<value_t, ReduceOperator>;
    value_t new_value = RAJA::hip::impl::warp_reduce<combiner_t>(value, ident);
    data.template assign_param<ParamId>(new_value);

    // execute enclosed statements, and mask off everyone but lane 0
    thread_active = threadIdx.x == 0;
    if (thread_active)
    {
      // Only update to new value on root thread
      data.template assign_param<ParamId>(new_value);
    }
    enclosed_stmts_t::exec(data, thread_active);
  }

  static inline LaunchDims calculateDimensions(Data const& data)
  {
    // combine with enclosed statements
    LaunchDims enclosed_dims = enclosed_stmts_t::calculateDimensions(data);
    return enclosed_dims;
  }
};


}  // namespace internal
}  // end namespace RAJA


#endif /* RAJA_policy_hip_kernel_Reduce_HPP */
