/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA workgroup Vtable.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_loop_WorkGroup_Vtable_HPP
#define RAJA_loop_WorkGroup_Vtable_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/loop/policy.hpp"

#include "RAJA/pattern/WorkGroup/Vtable.hpp"


namespace RAJA
{

namespace detail
{

/*!
 * Populate and return a Vtable object
 */
template < typename T, typename ... CallArgs >
inline Vtable<CallArgs...> get_Vtable(loop_work const&)
{
  return Vtable<CallArgs...>{
        &Vtable_move_construct<T, CallArgs...>,
        &Vtable_call<T, CallArgs...>,
        &Vtable_destroy<T, CallArgs...>,
        sizeof(T)
      };
}

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
