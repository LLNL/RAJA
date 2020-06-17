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

#include "RAJA/policy/loop/policy.hpp"

#include "RAJA/pattern/detail/WorkGroup.hpp"


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


/*!
 * Runs work in a storage container in order
 * and returns any per run resources
 */
template <typename ALLOCATOR_T,
          typename INDEX_T,
          typename ... Args>
struct WorkRunner<
        RAJA::loop_work,
        RAJA::ordered,
        ALLOCATOR_T,
        INDEX_T,
        Args...>
    : WorkRunnerForallOrdered<
        RAJA::loop_exec,
        RAJA::loop_work,
        RAJA::ordered,
        ALLOCATOR_T,
        INDEX_T,
        Args...>
{ };

/*!
 * Runs work in a storage container in reverse order
 * and returns any per run resources
 */
template <typename ALLOCATOR_T,
          typename INDEX_T,
          typename ... Args>
struct WorkRunner<
        RAJA::loop_work,
        RAJA::reverse_ordered,
        ALLOCATOR_T,
        INDEX_T,
        Args...>
    : WorkRunnerForallReverse<
        RAJA::loop_exec,
        RAJA::loop_work,
        RAJA::reverse_ordered,
        ALLOCATOR_T,
        INDEX_T,
        Args...>
{ };

}  // namespace detail

// RAJA_DECLARE_ALL_REDUCERS(seq_reduce, detail::ReduceSeq)

}  // namespace RAJA

#endif  // closing endif for header file include guard
