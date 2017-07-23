/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing forallN OpenMP constructs.
 *
 ******************************************************************************
 */

#ifndef RAJA_forallN_openmp_HPP
#define RAJA_forallN_openmp_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_OPENMP)

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
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

#include "RAJA/policy/openmp/policy.hpp"
#include "RAJA/internal/ForallNPolicy.hpp"
#include "RAJA/util/types.hpp"

#include <type_traits>
#include <omp.h>

namespace RAJA
{

/******************************************************************
 *  ForallN CUDA policies
 ******************************************************************/

namespace detail
{

struct ForallN_OMP_Parallel_Tag {
};

template <typename Iter>
using CollapsePolPair = ForallN_PolicyPair<omp_collapse_nowait_exec, Iter>;

template <typename T>
using no_const = typename std::remove_const<T>::type;

} // closing brace for namespace detail

template <typename NEXT = Execute>
struct OMP_Parallel {
  // Identify this policy
  using PolicyTag = detail::ForallN_OMP_Parallel_Tag;

  // The next nested-loop execution policy
  using NextPolicy = NEXT;
};

/******************************************************************
 *  ForallN collapse nowait execution templates
 ******************************************************************/

template <typename Iterable1, typename Iterable2, typename... PREST>
struct ForallN_Executor<false,
                        detail::CollapsePolPair<Iterable1>,
                        detail::CollapsePolPair<Iterable2>,
                        PREST...> {

  detail::CollapsePolPair<Iterable1> iset_i;
  detail::CollapsePolPair<Iterable2> iset_j;

  using NextExec = ForallN_Executor<false, PREST...>;
  NextExec next_exec;

  RAJA_INLINE
  constexpr ForallN_Executor(detail::CollapsePolPair<Iterable1> const &i,
                             detail::CollapsePolPair<Iterable2> const &j,
                             PREST const &... prest)
      : iset_i(i), iset_j(j), next_exec(prest...)
  {
  }

  template <typename BODY>
  RAJA_INLINE void operator()(BODY body) const
  {
    const auto begin_i = iset_i.begin();
    const auto size_i = iset_i.size();
    const auto begin_j = iset_j.begin();
    const auto size_j = iset_j.size();

    using detail::no_const;

    ForallN_PeelOuter<0, NextExec, BODY> outer(next_exec, body);

#if !defined(RAJA_COMPILER_MSVC)
#pragma omp for nowait collapse(2)
#else
#pragma omp for nowait
#endif
    for (no_const<decltype(size_i)> i = 0; i < size_i; ++i) {
      for (no_const<decltype(size_j)> j = 0; j < size_j; ++j) {
        outer(*(begin_i + i), *(begin_j + j));
      }
    }
  }
};

template <typename Iterable1,
          typename Iterable2,
          typename Iterable3,
          typename... PREST>
struct ForallN_Executor<false,
                        detail::CollapsePolPair<Iterable1>,
                        detail::CollapsePolPair<Iterable2>,
                        detail::CollapsePolPair<Iterable3>,
                        PREST...> {
  detail::CollapsePolPair<Iterable1> iset_i;
  detail::CollapsePolPair<Iterable2> iset_j;
  detail::CollapsePolPair<Iterable3> iset_k;

  using NextExec = ForallN_Executor<false, PREST...>;
  NextExec next_exec;

  RAJA_INLINE
  constexpr ForallN_Executor(
      detail::CollapsePolPair<Iterable1> const &i,
      detail::CollapsePolPair<Iterable2> const &j,
      detail::CollapsePolPair<Iterable3> const &k,
      PREST... prest)
    : iset_i(i), iset_j(j), iset_k(k), next_exec(prest...)
  {
  }

  template <typename BODY>
  RAJA_INLINE void operator()(BODY body) const
  {
    const auto begin_i = iset_i.begin();
    const auto size_i = iset_i.size();
    const auto begin_j = iset_j.begin();
    const auto size_j = iset_j.size();
    const auto begin_k = iset_k.begin();
    const auto size_k = iset_k.size();

    using detail::no_const;

    ForallN_PeelOuter<0, NextExec, BODY> outer(next_exec, body);

#if !defined(RAJA_COMPILER_MSVC)
#pragma omp for nowait collapse(3)
#else
#pragma omp for nowait
#endif
    for (no_const<decltype(size_i)> i = 0; i < size_i; ++i) {
      for (no_const<decltype(size_j)> j = 0; j < size_j; ++j) {
        for (no_const<decltype(size_k)> k = 0; k < size_k; ++k) {
          outer(*(begin_i + i), *(begin_j + j), *(begin_k + k));
        }
      }
    }
  }
};

/******************************************************************
 *  forallN_policy(), OpenMP Parallel Region execution
 ******************************************************************/

/*!
 * \brief Tiling policy front-end function.
 */
template <typename POLICY, typename BODY, typename... PARGS>
RAJA_INLINE void forallN_policy(detail::ForallN_OMP_Parallel_Tag,
                                BODY body,
                                PARGS... pargs)
{
  using NextPolicy = typename POLICY::NextPolicy;
  using NextPolicyTag = typename POLICY::NextPolicy::PolicyTag;

  #pragma omp parallel firstprivate(body)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(), body, pargs...);
  }
}

}  // namespace RAJA

#endif  // closing endif for if defined(RAJA_ENABLE_OPENMP)

#endif  // closing endif for header file include guard
