/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for routines used to manage
 *          memory for CUDA reductions and other operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/config.hpp"

#include <unordered_set>

#include "RAJA/util/mutex.hpp"
#include "RAJA/util/Allocator.hpp"

#if defined(RAJA_ENABLE_OPENMP) && !defined(_OPENMP)
#error RAJA configured with ENABLE_OPENMP, but OpenMP not supported by current compiler
#endif


namespace RAJA
{

namespace detail
{

static std::unordered_set<Allocator*> s_allocator_set;

#if defined(RAJA_ENABLE_OPENMP)
static omp::mutex s_mutex;
#endif

void add_allocator(Allocator* aloc)
{
#if defined(RAJA_ENABLE_OPENMP)
    lock_guard<omp::mutex> lock(s_mutex);
#endif

  auto res = s_allocator_set.emplace(aloc);
  if (!res.second) {
    RAJA_ABORT_OR_THROW("RAJA::detail::add_allocator allocator already added");
  }
}

void remove_allocator(Allocator* aloc)
{
#if defined(RAJA_ENABLE_OPENMP)
    lock_guard<omp::mutex> lock(s_mutex);
#endif

  auto it = s_allocator_set.find(aloc);
  if (it == s_allocator_set.end()) {
    RAJA_ABORT_OR_THROW("RAJA::detail::remove_allocator can not remove unknown allocator");
  }
  s_allocator_set.erase(it);
}

} /* namespace detail */


std::vector<Allocator*> get_allocators()
{
#if defined(RAJA_ENABLE_OPENMP)
    lock_guard<omp::mutex> lock(detail::s_mutex);
#endif
  using std::begin;
  using std::end;

  return std::vector<Allocator*>{ begin(detail::s_allocator_set),
                                  end(detail::s_allocator_set) };
}

}  // namespace RAJA
