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
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_forall_tbb_HPP
#define RAJA_forall_tbb_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_TBB)

#include <tbb/tbb.h>

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"
#include "RAJA/internal/fault_tolerance.hpp"
#include "RAJA/pattern/forall.hpp"
#include "RAJA/pattern/params/forall.hpp"
#include "RAJA/policy/tbb/policy.hpp"
#include "RAJA/util/types.hpp"


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

template <typename Iterable, typename Func, typename ForallParam>
RAJA_INLINE 
concepts::enable_if_t<
  resources::EventProxy<resources::Host>,
  expt::type_traits::is_ForallParamPack<ForallParam>,
  concepts::negate<expt::type_traits::is_ForallParamPack_empty<ForallParam>>
  >
forall_impl(resources::Host host_res,
            const tbb_for_dynamic& p,
            Iterable&& iter,
            Func&& loop_body,
            ForallParam f_params)
{
  using std::begin;
  using std::distance;
  using std::end;
  using brange = ::tbb::blocked_range<size_t>;
  auto b = begin(iter);
  size_t dist = std::abs(distance(begin(iter), end(iter)));

  expt::ParamMultiplexer::init<tbb_for_dynamic>(f_params);

  f_params = ::tbb::parallel_reduce(
      brange(0, dist, p.grain_size),

      f_params,

      [=](const brange& r, ForallParam fp) {
        using RAJA::internal::thread_privatize;
        auto privatizer = thread_privatize(loop_body);
        auto& body = privatizer.get_priv();
        for (auto i = r.begin(); i != r.end(); ++i)
          expt::invoke_body(fp, body, b[i]);
        return fp;
      },

      [](ForallParam lhs, ForallParam rhs) -> ForallParam {
        expt::ParamMultiplexer::combine<tbb_for_dynamic>(lhs, rhs);
        return lhs;
      }
  );

  expt::ParamMultiplexer::resolve<tbb_for_dynamic>(f_params);

  return resources::EventProxy<resources::Host>(host_res);
}

template <typename Iterable, typename Func, typename ForallParam>
RAJA_INLINE 
concepts::enable_if_t<
  resources::EventProxy<resources::Host>,
  expt::type_traits::is_ForallParamPack<ForallParam>,
  expt::type_traits::is_ForallParamPack_empty<ForallParam>
  >
forall_impl(resources::Host host_res,
            const tbb_for_dynamic& p,
            Iterable&& iter,
            Func&& loop_body,
            ForallParam)
{
  using std::begin;
  using std::distance;
  using std::end;
  using brange = ::tbb::blocked_range<size_t>;
  auto b = begin(iter);
  size_t dist = std::abs(distance(begin(iter), end(iter)));
  ::tbb::parallel_for(brange(0, dist, p.grain_size), [=](const brange& r) {
    using RAJA::internal::thread_privatize;
    auto privatizer = thread_privatize(loop_body);
    auto& body = privatizer.get_priv();
    for (auto i = r.begin(); i != r.end(); ++i)
      body(b[i]);
  });

  return resources::EventProxy<resources::Host>(host_res);
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

template <typename Iterable, typename Func, size_t ChunkSize, typename ForallParam>
RAJA_INLINE 
concepts::enable_if_t<
  resources::EventProxy<resources::Host>,
  expt::type_traits::is_ForallParamPack<ForallParam>,
  concepts::negate<expt::type_traits::is_ForallParamPack_empty<ForallParam>>
  >
forall_impl(resources::Host host_res,
            const tbb_for_static<ChunkSize>&,
            Iterable&& iter,
            Func&& loop_body,
            ForallParam f_params)
{
  using std::begin;
  using std::distance;
  using std::end;
  using brange = ::tbb::blocked_range<size_t>;
  auto b = begin(iter);
  size_t dist = std::abs(distance(begin(iter), end(iter)));

  expt::ParamMultiplexer::init<tbb_for_dynamic>(f_params);

  auto fp = ::tbb::parallel_reduce(
      brange(0, dist, ChunkSize),

      f_params,

      [=](const brange& r, ForallParam fp) {
        using RAJA::internal::thread_privatize;
        auto privatizer = thread_privatize(loop_body);
        auto& body = privatizer.get_priv();
        for (auto i = r.begin(); i != r.end(); ++i)
          expt::invoke_body(fp, body, b[i]);
        return fp;
      },

      [](ForallParam lhs, ForallParam rhs) -> ForallParam {
        expt::ParamMultiplexer::combine<tbb_for_dynamic>(lhs, rhs);
        return lhs;
      },
      tbb_static_partitioner{}

  );
  expt::ParamMultiplexer::combine<tbb_for_dynamic>(f_params, fp);

  expt::ParamMultiplexer::resolve<tbb_for_dynamic>(f_params);

  return resources::EventProxy<resources::Host>(host_res);
}

template <typename Iterable, typename Func, size_t ChunkSize, typename ForallParam>
RAJA_INLINE 
concepts::enable_if_t<
  resources::EventProxy<resources::Host>,
  expt::type_traits::is_ForallParamPack<ForallParam>,
  expt::type_traits::is_ForallParamPack_empty<ForallParam>
  >
forall_impl(resources::Host host_res,
            const tbb_for_static<ChunkSize>&,
            Iterable&& iter,
            Func&& loop_body,
            ForallParam)
{
  using std::begin;
  using std::distance;
  using std::end;
  using brange = ::tbb::blocked_range<size_t>;
  auto b = begin(iter);
  size_t dist = std::abs(distance(begin(iter), end(iter)));
  ::tbb::parallel_for(
      brange(0, dist, ChunkSize),
      [=](const brange& r) {
        using RAJA::internal::thread_privatize;
        auto privatizer = thread_privatize(loop_body);
        auto& body = privatizer.get_priv();
        for (auto i = r.begin(); i != r.end(); ++i)
          body(b[i]);
      },
      tbb_static_partitioner{});

  return resources::EventProxy<resources::Host>(host_res);
}

}  // namespace tbb
}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for if defined(RAJA_ENABLE_TBB)

#endif  // closing endif for header file include guard
