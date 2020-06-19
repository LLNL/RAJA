/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA WorkRunner class specializations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_hip_WorkGroup_WorkRunner_HPP
#define RAJA_hip_WorkGroup_WorkRunner_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/hip/policy.hpp"
#include "RAJA/policy/hip/MemUtils_HIP.hpp"

#include "RAJA/pattern/WorkGroup/WorkRunner.hpp"


namespace RAJA
{

namespace detail
{

/*!
 * Runs work in a storage container in order
 * and returns any per run resources
 */
template <size_t BLOCK_SIZE, bool Async,
          typename ALLOCATOR_T,
          typename INDEX_T,
          typename ... Args>
struct WorkRunner<
        RAJA::hip_work<BLOCK_SIZE, Async>,
        RAJA::ordered,
        ALLOCATOR_T,
        INDEX_T,
        Args...>
    : WorkRunnerForallOrdered<
        RAJA::hip_exec_async<BLOCK_SIZE>,
        RAJA::hip_work<BLOCK_SIZE, Async>,
        RAJA::ordered,
        ALLOCATOR_T,
        INDEX_T,
        Args...>
{
  using base = WorkRunnerForallOrdered<
        RAJA::hip_exec_async<BLOCK_SIZE>,
        RAJA::hip_work<BLOCK_SIZE, Async>,
        RAJA::ordered,
        ALLOCATOR_T,
        INDEX_T,
        Args...>;
  using base::base;
  using IndexType = INDEX_T;
  using per_run_storage = typename base::per_run_storage;

  template < typename WorkContainer >
  per_run_storage run(WorkContainer const& storage, Args... args) const
  {
    per_run_storage run_storage =
        base::run(storage, std::forward<Args>(args)...);

    IndexType num_loops = std::distance(std::begin(storage), std::end(storage));

    // Only synchronize if we had something to iterate over
    if (num_loops > 0 && BLOCK_SIZE > 0) {
      hipStream_t stream = 0; // TODO: coordinate with base to use same stream
      if (!Async) { RAJA::hip::synchronize(stream); }
    }

    return run_storage;
  }
};

/*!
 * Runs work in a storage container in reverse order
 * and returns any per run resources
 */
template <size_t BLOCK_SIZE, bool Async,
          typename ALLOCATOR_T,
          typename INDEX_T,
          typename ... Args>
struct WorkRunner<
        RAJA::hip_work<BLOCK_SIZE, Async>,
        RAJA::reverse_ordered,
        ALLOCATOR_T,
        INDEX_T,
        Args...>
    : WorkRunnerForallReverse<
        RAJA::hip_exec_async<BLOCK_SIZE>,
        RAJA::hip_work<BLOCK_SIZE, Async>,
        RAJA::reverse_ordered,
        ALLOCATOR_T,
        INDEX_T,
        Args...>
{
  using base = WorkRunnerForallReverse<
        RAJA::hip_exec_async<BLOCK_SIZE>,
        RAJA::hip_work<BLOCK_SIZE, Async>,
        RAJA::reverse_ordered,
        ALLOCATOR_T,
        INDEX_T,
        Args...>;
  using base::base;
  using IndexType = INDEX_T;
  using per_run_storage = typename base::per_run_storage;

  template < typename WorkContainer >
  per_run_storage run(WorkContainer const& storage, Args... args) const
  {
    per_run_storage run_storage =
        base::run(storage, std::forward<Args>(args)...);

    IndexType num_loops = std::distance(std::begin(storage), std::end(storage));

    // Only synchronize if we had something to iterate over
    if (num_loops > 0 && BLOCK_SIZE > 0) {
      hipStream_t stream = 0; // TODO: coordinate with base to use same stream
      if (!Async) { RAJA::hip::synchronize(stream); }
    }

    return run_storage;
  }
};

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
