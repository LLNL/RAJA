/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set and segment iteration
 *          template methods for sequential execution.
 *
 *          These methods should work on any platform.
 *
 *          Note: GNU compiler does not enforce sequential iterations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_forall_sequential_HPP
#define RAJA_forall_sequential_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/types.hpp"

#include "RAJA/policy/sequential/policy.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/pattern/detail/forall.hpp"

#include "RAJA/util/resource.hpp"

#include "RAJA/pattern/params/forall.hpp"

namespace RAJA
{
namespace policy
{
namespace sequential
{


//
//////////////////////////////////////////////////////////////////////
//
// The following function templates iterate over index set segments
// sequentially.  Segment execution is defined by segment
// execution policy template parameter.
//
//////////////////////////////////////////////////////////////////////
//

template <typename Iterable,
          typename Func,
          typename Resource,
          typename ForallParam>
RAJA_INLINE concepts::enable_if_t<
    resources::EventProxy<Resource>,
    expt::type_traits::is_ForallParamPack<ForallParam>,
    concepts::negate<expt::type_traits::is_ForallParamPack_empty<ForallParam>>>
forall_impl(Resource res,
            const seq_exec&,
            Iterable&& iter,
            Func&& body,
            ForallParam f_params)
{
  expt::ParamMultiplexer::init<seq_exec>(f_params);

  RAJA_EXTRACT_BED_IT(iter);

  for (decltype(distance_it) i = 0; i < distance_it; ++i)
  {
    expt::invoke_body(f_params, body, *(begin_it + i));
  }

  expt::ParamMultiplexer::resolve<seq_exec>(f_params);
  return resources::EventProxy<Resource>(res);
}

template <typename Iterable,
          typename Func,
          typename Resource,
          typename ForallParam>
RAJA_INLINE concepts::enable_if_t<
    resources::EventProxy<Resource>,
    expt::type_traits::is_ForallParamPack<ForallParam>,
    expt::type_traits::is_ForallParamPack_empty<ForallParam>>
forall_impl(Resource res,
            const seq_exec&,
            Iterable&& iter,
            Func&& body,
            ForallParam)
{
  RAJA_EXTRACT_BED_IT(iter);

  for (decltype(distance_it) i = 0; i < distance_it; ++i)
  {
    body(*(begin_it + i));
  }
  return resources::EventProxy<Resource>(res);
}

} // namespace sequential

} // namespace policy

} // namespace RAJA

#endif // closing endif for header file include guard
