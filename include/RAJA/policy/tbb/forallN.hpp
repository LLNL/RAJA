/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing forallN TBB constructs.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_forallN_tbb_HPP
#define RAJA_forallN_tbb_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_TBB)

#include "RAJA/internal/ForallNPolicy.hpp"
#include "RAJA/policy/tbb/policy.hpp"
#include "RAJA/util/types.hpp"

#include <tbb/tbb.h>
#include <cstddef>
#include <type_traits>

namespace RAJA
{

/******************************************************************
 *  ForallN TBB policies
 ******************************************************************/

namespace detail
{

struct ForallN_TBB_Parallel_Tag {
};

template <typename Iter, std::size_t GrainSize>
using TBBCollapsePolPair = ForallN_PolicyPair<tbb_for_static<GrainSize>, Iter>;

}  // closing brace for namespace detail

template <typename NEXT = Execute>
struct TBB_Parallel {
  // Identify this policy
  using PolicyTag = detail::ForallN_TBB_Parallel_Tag;

  // The next nested-loop execution policy
  using NextPolicy = NEXT;
};

/******************************************************************
 *  ForallN collapse nowait execution templates
 ******************************************************************/

template <typename Iterable1,
          typename Iterable2,
          std::size_t Grain1,
          std::size_t Grain2,
          typename... PREST>
struct ForallN_Executor<false,
                        detail::TBBCollapsePolPair<Iterable1, Grain1>,
                        detail::TBBCollapsePolPair<Iterable2, Grain2>,
                        PREST...> {

  detail::TBBCollapsePolPair<Iterable1, Grain1> iset_i;
  detail::TBBCollapsePolPair<Iterable2, Grain2> iset_j;

  using NextExec = ForallN_Executor<false, PREST...>;
  NextExec next_exec;

  RAJA_INLINE
  constexpr ForallN_Executor(
      detail::TBBCollapsePolPair<Iterable1, Grain1> const &i,
      detail::TBBCollapsePolPair<Iterable2, Grain2> const &j,
      PREST const &... prest)
      : iset_i(i), iset_j(j), next_exec(prest...)
  {
  }

  template <typename BODY>
  RAJA_INLINE void operator()(BODY body) const
  {
    ForallN_PeelOuter<0, NextExec, BODY> outer(next_exec, body);

    using brange = tbb::blocked_range2d<typename Iterable1::iterator,
                                        typename Iterable2::iterator>;
    tbb::parallel_for(brange(iset_i.begin(),
                             iset_i.end(),
                             Grain1,
                             iset_j.begin(),
                             iset_j.end(),
                             Grain2),
                      [=](const brange &r) {
                        for (auto i : r.rows()) {
                          for (auto j : r.cols()) {
                            outer(i, j);
                          }
                        }
                      });
  }
};

template <typename Iterable1,
          typename Iterable2,
          typename Iterable3,
          std::size_t Grain1,
          std::size_t Grain2,
          std::size_t Grain3,
          typename... PREST>
struct ForallN_Executor<false,
                        detail::TBBCollapsePolPair<Iterable1, Grain1>,
                        detail::TBBCollapsePolPair<Iterable2, Grain2>,
                        detail::TBBCollapsePolPair<Iterable3, Grain3>,
                        PREST...> {
  detail::TBBCollapsePolPair<Iterable1, Grain1> iset_i;
  detail::TBBCollapsePolPair<Iterable2, Grain2> iset_j;
  detail::TBBCollapsePolPair<Iterable3, Grain3> iset_k;

  using NextExec = ForallN_Executor<false, PREST...>;
  NextExec next_exec;

  RAJA_INLINE
  constexpr ForallN_Executor(
      detail::TBBCollapsePolPair<Iterable1, Grain1> const &i,
      detail::TBBCollapsePolPair<Iterable2, Grain2> const &j,
      detail::TBBCollapsePolPair<Iterable3, Grain3> const &k,
      PREST... prest)
      : iset_i(i), iset_j(j), iset_k(k), next_exec(prest...)
  {
  }

  template <typename BODY>
  RAJA_INLINE void operator()(BODY body) const
  {
    ForallN_PeelOuter<0, NextExec, BODY> outer(next_exec, body);

    using brange = tbb::blocked_range3d<typename Iterable1::iterator,
                                        typename Iterable2::iterator,
                                        typename Iterable3::iterator>;
    tbb::parallel_for(brange(iset_i.begin(),
                             iset_i.end(),
                             Grain1,
                             iset_j.begin(),
                             iset_j.end(),
                             Grain2,
                             iset_k.begin(),
                             iset_k.end(),
                             Grain3),
                      [=](const brange &r) {
                        for (auto i : r.pages()) {
                          for (auto j : r.rows()) {
                            for (auto k : r.cols()) {
                              outer(i, j, k);
                            }
                          }
                        }
                      });
  }
};

/******************************************************************
 *  forallN_policy(), TBB Parallel Region execution
 ******************************************************************/

/*!
 * \brief Tiling policy front-end function.
 */
template <typename POLICY, typename BODY, typename... PARGS>
RAJA_INLINE void forallN_policy(detail::ForallN_TBB_Parallel_Tag,
                                BODY body,
                                PARGS... pargs)
{
  using NextPolicy = typename POLICY::NextPolicy;
  using NextPolicyTag = typename POLICY::NextPolicy::PolicyTag;

  {
    forallN_policy<NextPolicy>(NextPolicyTag(), body, pargs...);
  }
}

}  // namespace RAJA

#endif  // closing endif for if defined(RAJA_ENABLE_TBB)

#endif  // closing endif for header file include guard
