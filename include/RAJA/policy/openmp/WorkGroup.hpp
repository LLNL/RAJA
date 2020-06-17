/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA WorkGroup templates for
 *          openmp execution.
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

#ifndef RAJA_openmp_WorkGroup_HPP
#define RAJA_openmp_WorkGroup_HPP

#include "RAJA/config.hpp"

#include "RAJA/internal/MemUtils_CPU.hpp"
#include "RAJA/policy/openmp/policy.hpp"
#include "RAJA/policy/loop/WorkGroup.hpp"

#include "RAJA/pattern/detail/WorkGroup.hpp"


namespace RAJA
{

namespace detail
{

/*!
* Populate and return a Vtable object
*/
template < typename T, typename ... CallArgs >
inline Vtable<CallArgs...> get_Vtable(omp_work const&)
{
  return get_Vtable<T, CallArgs...>(loop_work{});
}


/*!
 * Runs work in a storage container in order
 * and returns any per run resources
 */
template <typename ALLOCATOR_T,
          typename INDEX_T,
          typename ... Args>
struct WorkRunner<
        RAJA::omp_work,
        RAJA::ordered,
        ALLOCATOR_T,
        INDEX_T,
        Args...>
    : WorkRunnerForallOrdered<
        RAJA::omp_parallel_for_exec,
        RAJA::omp_work,
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
        RAJA::omp_work,
        RAJA::reverse_ordered,
        ALLOCATOR_T,
        INDEX_T,
        Args...>
    : WorkRunnerForallReverse<
        RAJA::omp_parallel_for_exec,
        RAJA::omp_work,
        RAJA::reverse_ordered,
        ALLOCATOR_T,
        INDEX_T,
        Args...>
{ };

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
