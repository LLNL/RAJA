//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __REDUCE_CONDITIONAL_LOOP_SEGMENTS_IMPL_HPP__
#define __REDUCE_CONDITIONAL_LOOP_SEGMENTS_IMPL_HPP__

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <vector>

template<typename EXEC_POL, bool USE_RESOURCE,
         typename PARAMETERS,
         typename SEGMENTS,
         typename WORKING_RES,
         typename... Args>
typename std::enable_if< USE_RESOURCE >::type call_kernel(SEGMENTS&& segs, PARAMETERS&& params, WORKING_RES& working_res, Args&&... args) {
  RAJA::kernel_param_resource<EXEC_POL>( segs, params, working_res, args...);
}

template<typename EXEC_POL, bool USE_RESOURCE,
         typename PARAMETERS,
         typename SEGMENTS,
         typename WORKING_RES,
         typename... Args>
typename std::enable_if< !USE_RESOURCE >::type call_kernel(SEGMENTS&& segs, PARAMETERS&& params, WORKING_RES&, Args&&... args) {
  RAJA::kernel_param<EXEC_POL>( segs, params, args...);
}

template <typename IDX_TYPE, typename EXEC_POLICY, typename REDUCE_POLICY,
          typename WORKING_RES, typename SEG_TYPE, bool USE_RESOURCE>
void KernelReduceConditionalLoopTestImpl(const SEG_TYPE& seg,
                                         WORKING_RES& working_res,
                                         camp::resources::Resource& erased_working_res)
{
  using std::begin;
  using std::end;
  using std::distance;
  auto begin_it = begin(seg);
  auto end_it = end(seg);
  auto distance_it = distance(begin_it, end_it);
  long N = (long)distance_it;

  RAJA::ReduceSum<REDUCE_POLICY, long> trip_count(0);

  for (int param = 0; param < 2; ++param) {

    trip_count.reset(0);

    call_kernel<EXEC_POLICY, USE_RESOURCE>(

        RAJA::make_tuple(seg),

        RAJA::make_tuple((bool)param),

        working_res,

        // This only gets executed if param==1
        [=] RAJA_HOST_DEVICE(IDX_TYPE) { trip_count += 2; },

        // This always gets executed
        [=] RAJA_HOST_DEVICE(IDX_TYPE) { trip_count += 1; });

    long result = (long)trip_count;

    ASSERT_EQ(result, N * (2 * param + 1));
  }
}

#endif  // __REDUCE_CONDITIONAL_LOOP_SEGMENTS_IMPL_HPP__
