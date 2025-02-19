/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA segment template methods for
 *          SIMD execution.
 *
 *          These methods should work on any platform. They make no
 *          asumptions about data alignment.
 *
 *          Note: Reduction operations should not be used with simd
 *          policies. Limited support.
 *
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_forall_simd_HPP
#define RAJA_forall_simd_HPP

#include "RAJA/config.hpp"

#include <iterator>
#include <type_traits>

#include "RAJA/util/types.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/policy/simd/policy.hpp"

#include "RAJA/pattern/params/forall.hpp"

namespace RAJA
{
namespace policy
{
namespace simd
{


template<typename Iterable, typename Func, typename ForallParam>
RAJA_INLINE concepts::enable_if_t<
    resources::EventProxy<resources::Host>,
    expt::type_traits::is_ForallParamPack<ForallParam>,
    concepts::negate<expt::type_traits::is_ForallParamPack_empty<ForallParam>>>
forall_impl(RAJA::resources::Host host_res,
            const simd_exec& RAJA_UNUSED_ARG(pol),
            Iterable&& iter,
            Func&& loop_body,
            ForallParam f_params)
{
  expt::ParamMultiplexer::params_init(seq_exec{}, f_params);

  auto begin    = std::begin(iter);
  auto end      = std::end(iter);
  auto distance = std::distance(begin, end);
  RAJA_SIMD
  for (decltype(distance) i = 0; i < distance; ++i)
  {
    expt::invoke_body(f_params, loop_body, *(begin + i));
  }

  expt::ParamMultiplexer::params_resolve(seq_exec{}, f_params);
  return RAJA::resources::EventProxy<resources::Host>(host_res);
}

template<typename Iterable, typename Func, typename ForallParam>
RAJA_INLINE concepts::enable_if_t<
    resources::EventProxy<resources::Host>,
    expt::type_traits::is_ForallParamPack<ForallParam>,
    expt::type_traits::is_ForallParamPack_empty<ForallParam>>
forall_impl(RAJA::resources::Host host_res,
            const simd_exec&,
            Iterable&& iter,
            Func&& loop_body,
            ForallParam)
{
  auto begin    = std::begin(iter);
  auto end      = std::end(iter);
  auto distance = std::distance(begin, end);
  RAJA_SIMD
  for (decltype(distance) i = 0; i < distance; ++i)
  {
    loop_body(*(begin + i));
  }

  return RAJA::resources::EventProxy<resources::Host>(host_res);
}

}  // namespace simd

}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for header file include guard
