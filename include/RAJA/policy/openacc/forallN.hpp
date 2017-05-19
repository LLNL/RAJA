/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing forallN Openacc constructs.
 *
 ******************************************************************************
 */

#ifndef RAJA_forallN_openacc_HXX__
#define RAJA_forallN_openacc_HXX__

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_OPENACC)

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

#include "RAJA/internal/ForallNPolicy.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/openacc/policy.hpp"

namespace RAJA
{

/******************************************************************
 *  ForallN CUDA policies
 ******************************************************************/

struct ACC_Parallel_Tag {
};
template <typename Config = acc::config<>, typename NEXT = Execute>
struct ACC_Parallel : public Config {
  using PolicyTag = ACC_Parallel_Tag;
  using NextPolicy = NEXT;
};

struct ACC_Kernels_Tag {
};
template <typename Config = acc::config<>, typename NEXT = Execute>
struct ACC_Kernels : public Config {
  using PolicyTag = ACC_Kernels_Tag;
  using NextPolicy = NEXT;
};

/******************************************************************
 *  forallN_policy(), Openacc Parallel Region execution
 ******************************************************************/

using namespace RAJA::acc;

template <typename Policy, typename Body, typename... PArgs>
RAJA_VERBOSE("\nacc parallel")
RAJA_INLINE When<Policy, no_ngangs, no_nworkers, no_nvectors> forallN_policy(
    ACC_Parallel_Tag,
    Body body,
    PArgs... pargs)
{
  using NextPolicy = typename Policy::NextPolicy;
  using NextPolicyTag = typename Policy::NextPolicy::PolicyTag;
#pragma acc parallel
  {
    forallN_policy<NextPolicy>(NextPolicyTag(),
                               std::forward<Body>(body),
                               std::forward<PArgs>(pargs)...);
  }
}

template <typename Policy, typename Body, typename... PArgs>
RAJA_VERBOSE("\nacc parallel nvectors")
RAJA_INLINE
    When<Policy, no_ngangs, no_nworkers, nvectors> forallN_policy(
        ACC_Parallel_Tag,
        Body body,
        PArgs... pargs)
{
  using NextPolicy = typename Policy::NextPolicy;
  using NextPolicyTag = typename Policy::NextPolicy::PolicyTag;
#pragma acc parallel vector_length(Policy::nvectors)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(),
                               std::forward<Body>(body),
                               std::forward<PArgs>(pargs)...);
  }
}

template <typename Policy, typename Body, typename... PArgs>
RAJA_VERBOSE("\nacc parallel nworkers")
RAJA_INLINE
    When<Policy, no_ngangs, nworkers, no_nvectors> forallN_policy(
        ACC_Parallel_Tag,
        Body body,
        PArgs... pargs)
{
  using NextPolicy = typename Policy::NextPolicy;
  using NextPolicyTag = typename Policy::NextPolicy::PolicyTag;
#pragma acc parallel num_workers(Policy::nworkers)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(),
                               std::forward<Body>(body),
                               std::forward<PArgs>(pargs)...);
  }
}

template <typename Policy, typename Body, typename... PArgs>
RAJA_VERBOSE("\nacc parallel nworkers nvectors")
RAJA_INLINE
    When<Policy, no_ngangs, nworkers, nvectors> forallN_policy(
        ACC_Parallel_Tag,
        Body body,
        PArgs... pargs)
{
  using NextPolicy = typename Policy::NextPolicy;
  using NextPolicyTag = typename Policy::NextPolicy::PolicyTag;
#pragma acc parallel num_workers(Policy::nworkers) \
    vector_length(Policy::nvectors)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(),
                               std::forward<Body>(body),
                               std::forward<PArgs>(pargs)...);
  }
}

template <typename Policy, typename Body, typename... PArgs>
RAJA_VERBOSE("\nacc parallel ngangs")
RAJA_INLINE
    When<Policy, ngangs, no_nworkers, no_nvectors> forallN_policy(
        ACC_Parallel_Tag,
        Body body,
        PArgs... pargs)
{
  using NextPolicy = typename Policy::NextPolicy;
  using NextPolicyTag = typename Policy::NextPolicy::PolicyTag;
#pragma acc parallel num_gangs(Policy::ngangs)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(),
                               std::forward<Body>(body),
                               std::forward<PArgs>(pargs)...);
  }
}

template <typename Policy, typename Body, typename... PArgs>
RAJA_VERBOSE("\nacc parallel ngangs nvectors")
RAJA_INLINE
    When<Policy, ngangs, no_nworkers, nvectors> forallN_policy(
        ACC_Parallel_Tag,
        Body body,
        PArgs... pargs)
{
  using NextPolicy = typename Policy::NextPolicy;
  using NextPolicyTag = typename Policy::NextPolicy::PolicyTag;
#pragma acc parallel num_gangs(Policy::ngangs) \
    vector_length(Policy::nvectors)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(),
                               std::forward<Body>(body),
                               std::forward<PArgs>(pargs)...);
  }
}

template <typename Policy, typename Body, typename... PArgs>
RAJA_VERBOSE("\nacc parallel ngangs nworkers")
RAJA_INLINE
    When<Policy, ngangs, nworkers, no_nvectors> forallN_policy(
        ACC_Parallel_Tag,
        Body body,
        PArgs... pargs)
{
  using NextPolicy = typename Policy::NextPolicy;
  using NextPolicyTag = typename Policy::NextPolicy::PolicyTag;
#pragma acc parallel num_gangs(Policy::ngangs) \
    num_workers(Policy::nworkers)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(),
                               std::forward<Body>(body),
                               std::forward<PArgs>(pargs)...);
  }
}

template <typename Policy, typename Body, typename... PArgs>
RAJA_VERBOSE("\nacc parallel ngangs nworkers nvectors")
RAJA_INLINE
    When<Policy, ngangs, nworkers, nvectors> forallN_policy(
        ACC_Parallel_Tag,
        Body body,
        PArgs... pargs)
{
  using NextPolicy = typename Policy::NextPolicy;
  using NextPolicyTag = typename Policy::NextPolicy::PolicyTag;
#pragma acc parallel num_gangs(Policy::ngangs) \
    num_workers(Policy::nworkers) vector_length(Policy::nvectors)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(),
                               std::forward<Body>(body),
                               std::forward<PArgs>(pargs)...);
  }
}


template <typename Policy, typename Body, typename... PArgs>
RAJA_VERBOSE("\nacc kernels")
RAJA_INLINE When<Policy, no_ngangs, no_nworkers, no_nvectors> forallN_policy(
    ACC_Kernels_Tag,
    Body body,
    PArgs... pargs)
{
  using NextPolicy = typename Policy::NextPolicy;
  using NextPolicyTag = typename Policy::NextPolicy::PolicyTag;
#pragma acc kernels
  {
    forallN_policy<NextPolicy>(NextPolicyTag(),
                               std::forward<Body>(body),
                               std::forward<PArgs>(pargs)...);
  }
}

template <typename Policy, typename Body, typename... PArgs>
RAJA_VERBOSE("\nacc kernels nvectors")
RAJA_INLINE
    When<Policy, no_ngangs, no_nworkers, nvectors> forallN_policy(
        ACC_Kernels_Tag,
        Body body,
        PArgs... pargs)
{
  using NextPolicy = typename Policy::NextPolicy;
  using NextPolicyTag = typename Policy::NextPolicy::PolicyTag;
#pragma acc kernels vector_length(Policy::nvectors)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(),
                               std::forward<Body>(body),
                               std::forward<PArgs>(pargs)...);
  }
}

template <typename Policy, typename Body, typename... PArgs>
RAJA_VERBOSE("\nacc kernels nworkers")
RAJA_INLINE
    When<Policy, no_ngangs, nworkers, no_nvectors> forallN_policy(
        ACC_Kernels_Tag,
        Body body,
        PArgs... pargs)
{
  using NextPolicy = typename Policy::NextPolicy;
  using NextPolicyTag = typename Policy::NextPolicy::PolicyTag;
#pragma acc kernels num_workers(Policy::nworkers)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(),
                               std::forward<Body>(body),
                               std::forward<PArgs>(pargs)...);
  }
}

template <typename Policy, typename Body, typename... PArgs>
RAJA_VERBOSE("\nacc kernels nworkers nvectors")
RAJA_INLINE
    When<Policy, no_ngangs, nworkers, nvectors> forallN_policy(
        ACC_Kernels_Tag,
        Body body,
        PArgs... pargs)
{
  using NextPolicy = typename Policy::NextPolicy;
  using NextPolicyTag = typename Policy::NextPolicy::PolicyTag;
#pragma acc kernels num_workers(Policy::nworkers) \
    vector_length(Policy::nvectors)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(),
                               std::forward<Body>(body),
                               std::forward<PArgs>(pargs)...);
  }
}

template <typename Policy, typename Body, typename... PArgs>
RAJA_VERBOSE("\nacc kernels ngangs")
RAJA_INLINE
    When<Policy, ngangs, no_nworkers, no_nvectors> forallN_policy(
        ACC_Kernels_Tag,
        Body body,
        PArgs... pargs)
{
  using NextPolicy = typename Policy::NextPolicy;
  using NextPolicyTag = typename Policy::NextPolicy::PolicyTag;
#pragma acc kernels num_gangs(Policy::ngangs)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(),
                               std::forward<Body>(body),
                               std::forward<PArgs>(pargs)...);
  }
}

template <typename Policy, typename Body, typename... PArgs>
RAJA_VERBOSE("\nacc kernels ngangs nvectors")
RAJA_INLINE
    When<Policy, ngangs, no_nworkers, nvectors> forallN_policy(
        ACC_Kernels_Tag,
        Body body,
        PArgs... pargs)
{
  using NextPolicy = typename Policy::NextPolicy;
  using NextPolicyTag = typename Policy::NextPolicy::PolicyTag;
#pragma acc kernels num_gangs(Policy::ngangs) \
    vector_length(Policy::nvectors)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(),
                               std::forward<Body>(body),
                               std::forward<PArgs>(pargs)...);
  }
}

template <typename Policy, typename Body, typename... PArgs>
RAJA_VERBOSE("\nacc kernels ngangs nworkers")
RAJA_INLINE
    When<Policy, ngangs, nworkers, no_nvectors> forallN_policy(
        ACC_Kernels_Tag,
        Body body,
        PArgs... pargs)
{
  using NextPolicy = typename Policy::NextPolicy;
  using NextPolicyTag = typename Policy::NextPolicy::PolicyTag;
#pragma acc kernels num_gangs(Policy::ngangs) \
    num_workers(Policy::nworkers)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(),
                               std::forward<Body>(body),
                               std::forward<PArgs>(pargs)...);
  }
}

template <typename Policy, typename Body, typename... PArgs>
RAJA_VERBOSE("\nacc kernels ngangs nworkers nvectors")
RAJA_INLINE
    When<Policy, ngangs, nworkers, nvectors> forallN_policy(
        ACC_Kernels_Tag,
        Body body,
        PArgs... pargs)
{
  using NextPolicy = typename Policy::NextPolicy;
  using NextPolicyTag = typename Policy::NextPolicy::PolicyTag;
#pragma acc kernels num_gangs(Policy::ngangs) \
    num_workers(Policy::nworkers) vector_length(Policy::nvectors)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(),
                               std::forward<Body>(body),
                               std::forward<PArgs>(pargs)...);
  }
}

}  // namespace RAJA

#endif  // closing endif for if defined(RAJA_ENABLE_OPENACC)

#endif  // closing endif for header file include guard
