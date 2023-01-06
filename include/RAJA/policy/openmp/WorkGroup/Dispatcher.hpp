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
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_openmp_WorkGroup_Dispatcher_HPP
#define RAJA_openmp_WorkGroup_Dispatcher_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/openmp/policy.hpp"

#include "RAJA/policy/loop/WorkGroup/Dispatcher.hpp"


namespace RAJA
{

namespace detail
{

/*!
* Populate and return a Dispatcher object
*/
template < typename T, typename Dispatcher_T >
inline const Dispatcher_T* get_Dispatcher(omp_work const&)
{
  return get_Dispatcher<T, Dispatcher_T>(loop_work{});
}

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
