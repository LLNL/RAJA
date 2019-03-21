/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for CUDA hyperplane executors.
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

#ifndef RAJA_policy_cuda_kernel_Hyperplane_HPP
#define RAJA_policy_cuda_kernel_Hyperplane_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "camp/camp.hpp"

#include "RAJA/pattern/kernel/Hyperplane.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

namespace RAJA
{
namespace internal
{


template <typename Data,
          camp::idx_t HpArgumentId,
          camp::idx_t... Args,
          typename... EnclosedStmts>
struct CudaStatementExecutor<Data,
                             statement::Hyperplane<HpArgumentId,
                                                   seq_exec,
                                                   ArgList<Args...>,
                                                   EnclosedStmts...>> {

  using stmt_list_t = StatementList<EnclosedStmts...>;
  using enclosed_stmts_t = CudaStatementListExecutor<Data, stmt_list_t>;

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    // compute Manhattan distance of iteration space to determine
    // as:  hp_len = l0 + l1 + l2 + ...
    int hp_len = segment_length<HpArgumentId>(data) +
                 VarOps::foldl(RAJA::operators::plus<int>(),
                               segment_length<Args>(data)...);

    int h_args = VarOps::foldl(RAJA::operators::plus<idx_t>(),
        camp::get<Args>(data.offset_tuple)...);

    // get length of i dimension
    auto i_len = segment_length<HpArgumentId>(data);


    /*
     * Execute the loop over hyperplanes
     *
     * We reject the iterations that lie outside of the specified rectangular
     * region we are sweeping.
     */
    for (int h = 0; h < hp_len; ++h) {

      // compute actual iterate for HpArgumentId
      // as:  i0 = h - (i1 + i2 + i3 + ...)
      idx_t i = h - h_args;

      // execute enclosed statements, masking off threads that are out of
      // bounds
      data.template assign_offset<HpArgumentId>(i);
      enclosed_stmts_t::exec(data, thread_active && (i >= 0 && i < i_len));
    }
  }



  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    return enclosed_stmts_t::calculateDimensions(data);
  }
};




}  // end namespace internal

}  // end namespace RAJA

#endif /* RAJA_pattern_kernel_HPP */
