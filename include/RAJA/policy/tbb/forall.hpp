/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set and segment iteration
 *          template methods for TBB.
 *
 *          These methods should work on any platform that supports TBB.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_forall_tbb_HPP
#define RAJA_forall_tbb_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_TBB)

#include "RAJA/util/types.hpp"

#include "RAJA/policy/tbb/policy.hpp"

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/pattern/forall.hpp"

#include <tbb/tbb.h>


namespace RAJA
{
#if TBB_VERSION_MAJOR >= 2017
using tbb_static_partitioner = tbb::static_partitioner;
#else
// @trws: This is not static, but it seems to be the least damaging option
// available pre-2017
using tbb_static_partitioner = tbb::auto_partitioner;
#endif

namespace policy
{
namespace tbb
{


/**
 * @brief TBB dynamic for implementation
 *
 * @param p tbb tag
 * @param iter any iterable
 * @param loop_body loop body
 *
 * @return None
 *
 *
 * This forall implements a TBB parallel_for loop over the specified iterable
 * using the dynamic loop scheduler and the grain size specified in the policy
 * argument.  This should be used for composable parallelism and increased work
 * stealing at the cost of initial start-up overhead for a top-level loop.
 */
template <typename Iterable, typename Func>
RAJA_INLINE void forall_impl(const tbb_for_dynamic& p,
                             Iterable&& iter,
                             Func&& loop_body)
{
  using std::begin;
  using std::end;
  using brange = ::tbb::blocked_range<decltype(iter.begin())>;
  ::tbb::parallel_for(brange(begin(iter), end(iter), p.grain_size),
                      [=](const brange& r) {
                        using RAJA::internal::thread_privatize;
                        auto privatizer = thread_privatize(loop_body);
                        auto body = privatizer.get_priv();
                        for (const auto& i : r)
                          body(i);
                      });
}

///
/// TBB parallel for static policy implementation
///

/**
 * @brief TBB static for implementation
 *
 * @param tbb_for_static tbb tag
 * @param iter any iterable
 * @param loop_body loop body
 *
 * @return None
 *
 * This forall implements a TBB parallel_for loop over the specified iterable
 * using the static loop scheduler and the grain size specified as a
 * compile-time constant in the policy argument.  This should be used for
 * OpenMP-like fast-launch well-balanced loops, or loops where the split between
 * threads must be maintained across multiple loops for correctness. NOTE: if
 * correctnes requires the per-thread mapping, you *must* use TBB 2017 or newer
 */
template <typename Iterable, typename Func, size_t ChunkSize>
RAJA_INLINE void forall_impl(const tbb_for_static<ChunkSize>&,
                             Iterable&& iter,
                             Func&& loop_body)
{
  using std::begin;
  using std::end;
  using brange = ::tbb::blocked_range<decltype(iter.begin())>;
  ::tbb::parallel_for(brange(begin(iter), end(iter), ChunkSize),
                      [=](const brange& r) {
                        using RAJA::internal::thread_privatize;
                        auto privatizer = thread_privatize(loop_body);
                        auto body = privatizer.get_priv();
                        for (const auto& i : r)
                          body(i);
                      },
                      tbb_static_partitioner{});
}

}  // closing brace for tbb namespace
}  // closing brace for policy namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for if defined(RAJA_ENABLE_TBB)

#endif  // closing endif for header file include guard
