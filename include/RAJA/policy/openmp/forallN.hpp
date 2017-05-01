/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing forallN OpenMP constructs.
 *
 ******************************************************************************
 */

#ifndef RAJA_forallN_openmp_HXX__
#define RAJA_forallN_openmp_HXX__

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

#include "RAJA/util/types.hpp"
#include "RAJA/internal/ForallNPolicy.hpp"

#include "RAJA/policy/openmp/policy.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace RAJA
{

/******************************************************************
 *  ForallN CUDA policies
 ******************************************************************/

struct ForallN_OMP_Parallel_Tag {
};
template <typename NEXT = Execute>
struct OMP_Parallel {
  // Identify this policy
  typedef ForallN_OMP_Parallel_Tag PolicyTag;

  // The next nested-loop execution policy
  typedef NEXT NextPolicy;
};

/******************************************************************
 *  ForallN collapse nowait policies
 ******************************************************************/

struct omp_collapse_nowait_exec {
};

template <typename... PREST>
struct ForallN_Executor<ForallN_PolicyPair<omp_collapse_nowait_exec,
                                           RangeSegment>,
                        ForallN_PolicyPair<omp_collapse_nowait_exec,
                                           RangeSegment>,
                        PREST...> {
  ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment> iset_i;
  ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment> iset_j;

  typedef ForallN_Executor<PREST...> NextExec;
  NextExec next_exec;

  RAJA_INLINE
  constexpr ForallN_Executor(
      ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment> const &iseti_,
      ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment> const &isetj_,
      PREST const &... prest)
      : iset_i(iseti_), iset_j(isetj_), next_exec(prest...)
  {
  }

  template <typename BODY>
  RAJA_INLINE void operator()(BODY body) const
  {
    int begin_i = iset_i.getBegin();
    int begin_j = iset_j.getBegin();
    int end_i = iset_i.getEnd();
    int end_j = iset_j.getEnd();

    ForallN_PeelOuter<NextExec, BODY> outer(next_exec, body);

#if !defined(RAJA_COMPILER_MSVC)
#pragma omp for nowait collapse(2)
#else
#pragma omp for nowait
#endif
    for (int i = begin_i; i < end_i; ++i) {
      for (int j = begin_j; j < end_j; ++j) {
        outer(i, j);
      }
    }
  }
};

template <typename... PREST>
struct ForallN_Executor<ForallN_PolicyPair<omp_collapse_nowait_exec,
                                           RangeSegment>,
                        ForallN_PolicyPair<omp_collapse_nowait_exec,
                                           RangeSegment>,
                        ForallN_PolicyPair<omp_collapse_nowait_exec,
                                           RangeSegment>,
                        PREST...> {
  ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment> iset_i;
  ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment> iset_j;
  ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment> iset_k;

  typedef ForallN_Executor<PREST...> NextExec;
  NextExec next_exec;

  RAJA_INLINE
  constexpr ForallN_Executor(
      ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment> const &iseti_,
      ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment> const &isetj_,
      ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment> const &isetk_,
      PREST... prest)
      : iset_i(iseti_), iset_j(isetj_), iset_k(isetk_), next_exec(prest...)
  {
  }

  template <typename BODY>
  RAJA_INLINE void operator()(BODY body) const
  {
    int begin_i = iset_i.getBegin();
    int begin_j = iset_j.getBegin();
    int begin_k = iset_k.getBegin();
    int end_i = iset_i.getEnd();
    int end_j = iset_j.getEnd();
    int end_k = iset_k.getEnd();

    ForallN_PeelOuter<NextExec, BODY> outer(next_exec, body);

#if !defined(RAJA_COMPILER_MSVC)
#pragma omp for nowait collapse(3)
#else
#pragma omp for nowait
#endif
    for (int i = begin_i; i < end_i; ++i) {
      for (int j = begin_j; j < end_j; ++j) {
        for (int k = begin_k; k < end_k; ++k) {
          outer(i, j, k);
        }
      }
    }
  }
};

/*
 * Collapse RangeStrideSegments
 */
template <typename... PREST>
struct ForallN_Executor<ForallN_PolicyPair<omp_collapse_nowait_exec,
                                           RangeStrideSegment>,
                        ForallN_PolicyPair<omp_collapse_nowait_exec,
                                           RangeStrideSegment>,
                        PREST...> {
  ForallN_PolicyPair<omp_collapse_nowait_exec, RangeStrideSegment> iset_i;
  ForallN_PolicyPair<omp_collapse_nowait_exec, RangeStrideSegment> iset_j;

  typedef ForallN_Executor<PREST...> NextExec;
  NextExec next_exec;

  RAJA_INLINE
  constexpr ForallN_Executor(
      ForallN_PolicyPair<omp_collapse_nowait_exec, RangeStrideSegment> const &iseti_,
      ForallN_PolicyPair<omp_collapse_nowait_exec, RangeStrideSegment> const &isetj_,
      PREST const &... prest)
      : iset_i(iseti_), iset_j(isetj_), next_exec(prest...)
  {
  }

  template <typename BODY>
  RAJA_INLINE void operator()(BODY body) const
  {
    int begin_i = iset_i.getBegin();
    int begin_j = iset_j.getBegin();
    int end_i = iset_i.getEnd();
    int end_j = iset_j.getEnd();
    int stride_i = iset_i.getStride();
    int stride_j = iset_j.getStride();


    ForallN_PeelOuter<NextExec, BODY> outer(next_exec, body);

#if !defined(RAJA_COMPILER_MSVC)
#pragma omp for nowait collapse(2)
#else
#pragma omp for nowait
#endif
    for (int i = begin_i; i < end_i; i+=stride_i) {
      for (int j = begin_j; j < end_j; j+=stride_j) {
        outer(i, j);
      }
    }
  }
};

template <typename... PREST>
struct ForallN_Executor<ForallN_PolicyPair<omp_collapse_nowait_exec,
                                           RangeStrideSegment>,
                        ForallN_PolicyPair<omp_collapse_nowait_exec,
                                           RangeStrideSegment>,
                        ForallN_PolicyPair<omp_collapse_nowait_exec,
                                           RangeStrideSegment>,
                        PREST...> {
  ForallN_PolicyPair<omp_collapse_nowait_exec, RangeStrideSegment> iset_i;
  ForallN_PolicyPair<omp_collapse_nowait_exec, RangeStrideSegment> iset_j;
  ForallN_PolicyPair<omp_collapse_nowait_exec, RangeStrideSegment> iset_k;

  typedef ForallN_Executor<PREST...> NextExec;
  NextExec next_exec;

  RAJA_INLINE
  constexpr ForallN_Executor(
      ForallN_PolicyPair<omp_collapse_nowait_exec, RangeStrideSegment> const &iseti_,
      ForallN_PolicyPair<omp_collapse_nowait_exec, RangeStrideSegment> const &isetj_,
      ForallN_PolicyPair<omp_collapse_nowait_exec, RangeStrideSegment> const &isetk_,
      PREST... prest)
      : iset_i(iseti_), iset_j(isetj_), iset_k(isetk_), next_exec(prest...)
  {
  }

  template <typename BODY>
  RAJA_INLINE void operator()(BODY body) const
  {
    int begin_i = iset_i.getBegin();
    int begin_j = iset_j.getBegin();
    int begin_k = iset_k.getBegin();
    int end_i = iset_i.getEnd();
    int end_j = iset_j.getEnd();
    int end_k = iset_k.getEnd();
    int stride_i = iset_i.getStride();
    int stride_j = iset_j.getStride();
    int stride_k = iset_k.getStride();

    ForallN_PeelOuter<NextExec, BODY> outer(next_exec, body);

#if !defined(RAJA_COMPILER_MSVC)
#pragma omp for nowait collapse(3)
#else
#pragma omp for nowait
#endif
    for (int i = begin_i; i < end_i; i+=stride_i) {
      for (int j = begin_j; j < end_j; j+=stride_j) {
        for (int k = begin_k; k < end_k; k+=stride_k) {
          outer(i, j, k);
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
RAJA_INLINE void forallN_policy(ForallN_OMP_Parallel_Tag,
                                BODY body,
                                PARGS... pargs)
{
  typedef typename POLICY::NextPolicy NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

#pragma omp parallel firstprivate(body)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(), body, pargs...);
  }
}

}  // namespace RAJA

#endif  // closing endif for if defined(RAJA_ENABLE_OPENMP)

#endif  // closing endif for header file include guard
