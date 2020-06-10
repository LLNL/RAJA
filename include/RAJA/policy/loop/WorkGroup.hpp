/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA WorkGroup templates for
 *          loop execution.
 *
 *          These methods should work on any platform.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_loop_WorkGroup_HPP
#define RAJA_loop_WorkGroup_HPP

#include "RAJA/config.hpp"

#include "RAJA/internal/MemUtils_CPU.hpp"

#include "RAJA/pattern/detail/WorkGroup.hpp"
#include "RAJA/pattern/WorkGroup.hpp"

#include "RAJA/policy/loop/policy.hpp"

namespace RAJA
{

namespace detail
{

/*!
 * Populate and return a Vtable object
 */
template < typename T, typename ... CallArgs >
Vtable<CallArgs...> get_Vtable_impl(loop_work const&)
{
  return Vtable<CallArgs...>{
        &Vtable_move_construct<T, CallArgs...>,
        &Vtable_call<T, CallArgs...>,
        &Vtable_destroy<T, CallArgs...>
      };
}

}  // namespace detail

// RAJA_DECLARE_ALL_REDUCERS(seq_reduce, detail::ReduceSeq)

}  // namespace RAJA

#endif  // closing endif for header file include guard
