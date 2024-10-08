/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for SYCL kernel conditional methods.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_sycl_kernel_Conditional_HPP
#define RAJA_policy_sycl_kernel_Conditional_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/kernel/Conditional.hpp"

#include "RAJA/policy/sycl/kernel/internal.hpp"

namespace RAJA
{
namespace internal
{


template <typename Data,
          typename Conditional,
          typename... EnclosedStmts,
          typename Types>
struct SyclStatementExecutor<Data,
                             statement::If<Conditional, EnclosedStmts...>,
                             Types> {

  using stmt_list_t = StatementList<EnclosedStmts...>;
  using enclosed_stmts_t = SyclStatementListExecutor<Data, stmt_list_t, Types>;


  static
  inline
  RAJA_DEVICE
  void exec(Data &data, cl::sycl::nd_item<3> item, bool thread_active)
  {
    if (Conditional::eval(data)) {

      // execute enclosed statements
      enclosed_stmts_t::exec(data, item, thread_active);
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


#endif
