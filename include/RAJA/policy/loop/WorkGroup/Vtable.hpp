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
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
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
template < typename T, typename Vtable_T >
inline const Vtable_T* get_Vtable(loop_work const&)
{
  static Vtable_T vtable{
        &Vtable_T::template move_construct_destroy<T>,
        &Vtable_T::template host_call<T>,
        &Vtable_T::template destroy<T>,
        sizeof(T)
      };
  return &vtable;
}

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
