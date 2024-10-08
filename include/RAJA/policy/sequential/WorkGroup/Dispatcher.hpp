/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA workgroup Dispatcher.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_sequential_WorkGroup_Dispatcher_HPP
#define RAJA_sequential_WorkGroup_Dispatcher_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/sequential/policy.hpp"

#include "RAJA/pattern/WorkGroup/Dispatcher.hpp"


namespace RAJA
{

namespace detail
{

/*!
 * Populate and return a Dispatcher object
 */
template <typename T, typename Dispatcher_T>
inline const Dispatcher_T* get_Dispatcher(seq_work const&)
{
  static Dispatcher_T dispatcher {Dispatcher_T::template makeDispatcher<T>()};
  return &dispatcher;
}

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
