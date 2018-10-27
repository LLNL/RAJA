/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for CUDA shared memory window executors.
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


#ifndef RAJA_policy_cuda_kernel_ShmemWindow_HPP
#define RAJA_policy_cuda_kernel_ShmemWindow_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/kernel/ShmemWindow.hpp"
#include "RAJA/policy/cuda/kernel/internal.hpp"

namespace RAJA
{
namespace internal
{


template <typename Data, typename... EnclosedStmts>
struct CudaStatementExecutor<Data,
                             statement::SetShmemWindow<EnclosedStmts...>> {

  using stmt_list_t = StatementList<EnclosedStmts...>;
  using enclosed_stmts_t = CudaStatementListExecutor<Data, stmt_list_t>;


  static
  inline
  RAJA_DEVICE
  void exec(Data &data)
  {

    // Call setWindow on all of our shmem objects
    RAJA::internal::shmem_set_windows(data.param_tuple,
                                      data.get_minimum_index_tuple());

    // execute enclosed statements
    enclosed_stmts_t::exec(data);
  }




  inline
  static
  LaunchDims calculateDimensions(Data const &data)
  {
    return enclosed_stmts_t::calculateDimensions(data);
  }
};


}  // namespace internal
}  // end namespace RAJA


#endif
