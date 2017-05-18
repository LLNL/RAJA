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

#include "RAJA/util/types.hpp"
#include "RAJA/internal/ForallNPolicy.hpp"

#include "RAJA/policy/openacc/policy.hpp"

#ifdef RAJA_ENABLE_VERBOSE
#define RAJA_VERBOSE(A) [[deprecated(A)]]
#else
#define RAJA_VERBOSE(A)
#endif

namespace RAJA
{

/******************************************************************
 *  ForallN CUDA policies
 ******************************************************************/

struct ForallN_ACC_Parallel_Tag {};
template <typename Config = acc::config<>, typename NEXT = Execute>
struct ACC_Parallel : public Config {
  using PolicyTag = ForallN_ACC_Parallel_Tag;
  using NextPolicy = NEXT;
};

struct ForallN_ACC_Kernels_Tag {};
template <typename Config = acc::config<>, typename NEXT = Execute>
struct ACC_Kernels : public Config {
  using PolicyTag = ForallN_ACC_Kernels_Tag;
  using NextPolicy = NEXT;
};

/******************************************************************
 *  forallN_policy(), Openacc Parallel Region execution
 ******************************************************************/

template <typename POLICY, typename BODY, typename... PARGS>
RAJA_VERBOSE("\nacc parallel") RAJA_INLINE
typename std::enable_if<
  !acc::has::num_gangs<POLICY>::value&&
  !acc::has::num_workers<POLICY>::value&&
  !acc::has::num_vectors<POLICY>::value>::type
forallN_policy(ForallN_ACC_Parallel_Tag, BODY body, PARGS... pargs) {
  using NextPolicy = typename POLICY::NextPolicy;
  using NextPolicyTag = typename POLICY::NextPolicy::PolicyTag;
  #pragma acc parallel
  {
    forallN_policy<NextPolicy>(NextPolicyTag(), std::forward<BODY>(body), std::forward<PARGS>(pargs)...);
  }
}
template <typename POLICY, typename BODY, typename... PARGS>
RAJA_VERBOSE("\nacc kernels") RAJA_INLINE
typename std::enable_if<
  !acc::has::num_gangs<POLICY>::value&&
  !acc::has::num_workers<POLICY>::value&&
  !acc::has::num_vectors<POLICY>::value>::type
forallN_policy(ForallN_ACC_Kernels_Tag, BODY body, PARGS... pargs) {
  using NextPolicy = typename POLICY::NextPolicy;
  using NextPolicyTag = typename POLICY::NextPolicy::PolicyTag;
  #pragma acc kernels
  {
    forallN_policy<NextPolicy>(NextPolicyTag(), std::forward<BODY>(body), std::forward<PARGS>(pargs)...);
  }
}

template <typename POLICY, typename BODY, typename... PARGS>
RAJA_VERBOSE("\nacc parallel num_gangs") RAJA_INLINE
typename std::enable_if<
  acc::has::num_gangs<POLICY>::value&&
  !acc::has::num_workers<POLICY>::value&&
  !acc::has::num_vectors<POLICY>::value>::type
forallN_policy(ForallN_ACC_Parallel_Tag, BODY body, PARGS... pargs) {
  using NextPolicy = typename POLICY::NextPolicy;
  using NextPolicyTag = typename POLICY::NextPolicy::PolicyTag;
  #pragma acc parallel num_gangs(POLICY::num_gangs)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(), std::forward<BODY>(body), std::forward<PARGS>(pargs)...);
  }
}
template <typename POLICY, typename BODY, typename... PARGS>
RAJA_VERBOSE("\nacc kernels num_gangs") RAJA_INLINE
typename std::enable_if<
  acc::has::num_gangs<POLICY>::value&&
  !acc::has::num_workers<POLICY>::value&&
  !acc::has::num_vectors<POLICY>::value>::type
forallN_policy(ForallN_ACC_Kernels_Tag, BODY body, PARGS... pargs) {
  using NextPolicy = typename POLICY::NextPolicy;
  using NextPolicyTag = typename POLICY::NextPolicy::PolicyTag;
  #pragma acc kernels num_gangs(POLICY::num_gangs)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(), std::forward<BODY>(body), std::forward<PARGS>(pargs)...);
  }
}

template <typename POLICY, typename BODY, typename... PARGS>
RAJA_VERBOSE("\nacc parallel num_workers") RAJA_INLINE
typename std::enable_if<
  !acc::has::num_gangs<POLICY>::value&&
  acc::has::num_workers<POLICY>::value&&
  !acc::has::num_vectors<POLICY>::value>::type
forallN_policy(ForallN_ACC_Parallel_Tag, BODY body, PARGS... pargs) {
  using NextPolicy = typename POLICY::NextPolicy;
  using NextPolicyTag = typename POLICY::NextPolicy::PolicyTag;
  #pragma acc parallel num_workers(POLICY::num_workers)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(), std::forward<BODY>(body), std::forward<PARGS>(pargs)...);
  }
}
template <typename POLICY, typename BODY, typename... PARGS>
RAJA_VERBOSE("\nacc kernels num_workers") RAJA_INLINE
typename std::enable_if<
  !acc::has::num_gangs<POLICY>::value&&
  acc::has::num_workers<POLICY>::value&&
  !acc::has::num_vectors<POLICY>::value>::type
forallN_policy(ForallN_ACC_Kernels_Tag, BODY body, PARGS... pargs) {
  using NextPolicy = typename POLICY::NextPolicy;
  using NextPolicyTag = typename POLICY::NextPolicy::PolicyTag;
  #pragma acc kernels num_workers(POLICY::num_workers)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(), std::forward<BODY>(body), std::forward<PARGS>(pargs)...);
  }
}

template <typename POLICY, typename BODY, typename... PARGS>
RAJA_VERBOSE("\nacc parallel vector_length") RAJA_INLINE
typename std::enable_if<
  !acc::has::num_gangs<POLICY>::value&&
  !acc::has::num_workers<POLICY>::value&&
  acc::has::num_vectors<POLICY>::value>::type
forallN_policy(ForallN_ACC_Parallel_Tag, BODY body, PARGS... pargs) {
  using NextPolicy = typename POLICY::NextPolicy;
  using NextPolicyTag = typename POLICY::NextPolicy::PolicyTag;
  #pragma acc parallel vector_length(POLICY::num_vectors)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(), std::forward<BODY>(body), std::forward<PARGS>(pargs)...);
  }
}
template <typename POLICY, typename BODY, typename... PARGS>
RAJA_VERBOSE("\nacc kernels vector_length") RAJA_INLINE
typename std::enable_if<
  !acc::has::num_gangs<POLICY>::value&&
  !acc::has::num_workers<POLICY>::value&&
  acc::has::num_vectors<POLICY>::value>::type
forallN_policy(ForallN_ACC_Kernels_Tag, BODY body, PARGS... pargs) {
  using NextPolicy = typename POLICY::NextPolicy;
  using NextPolicyTag = typename POLICY::NextPolicy::PolicyTag;
  #pragma acc kernels vector_length(POLICY::num_vectors)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(), std::forward<BODY>(body), std::forward<PARGS>(pargs)...);
  }
}


template <typename POLICY, typename BODY, typename... PARGS>
RAJA_VERBOSE("\nacc parallel num_gangs num_workers") RAJA_INLINE
typename std::enable_if<
  acc::has::num_gangs<POLICY>::value&&
  acc::has::num_workers<POLICY>::value&&
  !acc::has::num_vectors<POLICY>::value>::type
forallN_policy(ForallN_ACC_Parallel_Tag, BODY body, PARGS... pargs) {
  using NextPolicy = typename POLICY::NextPolicy;
  using NextPolicyTag = typename POLICY::NextPolicy::PolicyTag;
  #pragma acc parallel num_gangs(POLICY::num_gangs) num_workers(POLICY::num_workers)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(), std::forward<BODY>(body), std::forward<PARGS>(pargs)...);
  }
}
template <typename POLICY, typename BODY, typename... PARGS>
RAJA_VERBOSE("\nacc kernels num_gangs num_workers") RAJA_INLINE
typename std::enable_if<
  acc::has::num_gangs<POLICY>::value&&
  acc::has::num_workers<POLICY>::value&&
  !acc::has::num_vectors<POLICY>::value>::type
forallN_policy(ForallN_ACC_Kernels_Tag, BODY body, PARGS... pargs) {
  using NextPolicy = typename POLICY::NextPolicy;
  using NextPolicyTag = typename POLICY::NextPolicy::PolicyTag;
  #pragma acc kernels num_gangs(POLICY::num_gangs) num_workers(POLICY::num_workers)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(), std::forward<BODY>(body), std::forward<PARGS>(pargs)...);
  }
}

template <typename POLICY, typename BODY, typename... PARGS>
RAJA_VERBOSE("\nacc parallel num_gangs vector_length") RAJA_INLINE
typename std::enable_if<
  acc::has::num_gangs<POLICY>::value&&
  !acc::has::num_workers<POLICY>::value&&
  acc::has::num_vectors<POLICY>::value>::type
forallN_policy(ForallN_ACC_Parallel_Tag, BODY body, PARGS... pargs) {
  using NextPolicy = typename POLICY::NextPolicy;
  using NextPolicyTag = typename POLICY::NextPolicy::PolicyTag;
  #pragma acc parallel num_gangs(POLICY::num_gangs) vector_length(POLICY::num_vectors)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(), std::forward<BODY>(body), std::forward<PARGS>(pargs)...);
  }
}
template <typename POLICY, typename BODY, typename... PARGS>
RAJA_VERBOSE("\nacc kernels num_gangs vector_length") RAJA_INLINE
typename std::enable_if<
  acc::has::num_gangs<POLICY>::value&&
  !acc::has::num_workers<POLICY>::value&&
  acc::has::num_vectors<POLICY>::value>::type
forallN_policy(ForallN_ACC_Kernels_Tag, BODY body, PARGS... pargs) {
  using NextPolicy = typename POLICY::NextPolicy;
  using NextPolicyTag = typename POLICY::NextPolicy::PolicyTag;
  #pragma acc kernels num_gangs(POLICY::num_gangs) vector_length(POLICY::num_vectors)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(), std::forward<BODY>(body), std::forward<PARGS>(pargs)...);
  }
}

template <typename POLICY, typename BODY, typename... PARGS>
RAJA_VERBOSE("\nacc parallel num_workers vector_length") RAJA_INLINE
typename std::enable_if<
  !acc::has::num_gangs<POLICY>::value&&
  acc::has::num_workers<POLICY>::value&&
  acc::has::num_vectors<POLICY>::value>::type
forallN_policy(ForallN_ACC_Parallel_Tag, BODY body, PARGS... pargs) {
  using NextPolicy = typename POLICY::NextPolicy;
  using NextPolicyTag = typename POLICY::NextPolicy::PolicyTag;
  #pragma acc parallel num_workers(POLICY::num_workers) vector_length(POLICY::num_vectors)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(), std::forward<BODY>(body), std::forward<PARGS>(pargs)...);
  }
}
template <typename POLICY, typename BODY, typename... PARGS>
RAJA_VERBOSE("\nacc kernels num_workers vector_length") RAJA_INLINE
typename std::enable_if<
  !acc::has::num_gangs<POLICY>::value&&
  acc::has::num_workers<POLICY>::value&&
  acc::has::num_vectors<POLICY>::value>::type
forallN_policy(ForallN_ACC_Kernels_Tag, BODY body, PARGS... pargs) {
  using NextPolicy = typename POLICY::NextPolicy;
  using NextPolicyTag = typename POLICY::NextPolicy::PolicyTag;
  #pragma acc kernels num_workers(POLICY::num_workers) vector_length(POLICY::num_vectors)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(), std::forward<BODY>(body), std::forward<PARGS>(pargs)...);
  }
}

template <typename POLICY, typename BODY, typename... PARGS>
RAJA_VERBOSE("\nacc parallel num_gangs num_workers vector_length") RAJA_INLINE
typename std::enable_if<
  acc::has::num_gangs<POLICY>::value&&
  acc::has::num_workers<POLICY>::value&&
  acc::has::num_vectors<POLICY>::value>::type
forallN_policy(ForallN_ACC_Parallel_Tag, BODY body, PARGS... pargs) {
  using NextPolicy = typename POLICY::NextPolicy;
  using NextPolicyTag = typename POLICY::NextPolicy::PolicyTag;
  #pragma acc parallel num_gangs(POLICY::num_gangs) num_workers(POLICY::num_workers) vector_length(POLICY::num_vectors)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(), std::forward<BODY>(body), std::forward<PARGS>(pargs)...);
  }
}
template <typename POLICY, typename BODY, typename... PARGS>
RAJA_VERBOSE("\nacc kernels num_gangs num_workers vector_length") RAJA_INLINE
typename std::enable_if<
  acc::has::num_gangs<POLICY>::value&&
  acc::has::num_workers<POLICY>::value&&
  acc::has::num_vectors<POLICY>::value>::type
forallN_policy(ForallN_ACC_Kernels_Tag, BODY body, PARGS... pargs) {
  using NextPolicy = typename POLICY::NextPolicy;
  using NextPolicyTag = typename POLICY::NextPolicy::PolicyTag;
  #pragma acc kernels num_gangs(POLICY::num_gangs) num_workers(POLICY::num_workers) vector_length(POLICY::num_vectors)
  {
    forallN_policy<NextPolicy>(NextPolicyTag(), std::forward<BODY>(body), std::forward<PARGS>(pargs)...);
  }
}

}  // namespace RAJA

#undef RAJA_VERBOSE

#endif  // closing endif for if defined(RAJA_ENABLE_OPENACC)

#endif  // closing endif for header file include guard
