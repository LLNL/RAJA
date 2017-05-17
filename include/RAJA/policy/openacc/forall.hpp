/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set and segment iteration
 *          template methods for OpenACC.
 *
 *          These methods should work on any platform that supports OpenACC.
 *
 ******************************************************************************
 */

#ifndef RAJA_forall_openacc_HXX
#define RAJA_forall_openacc_HXX

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

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"

#include "RAJA/policy/openacc/policy.hpp"

#include <iostream>

#if defined(_OPENACC)
#include <openacc.h>
#endif


namespace RAJA
{

namespace impl
{

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<!acc::has::num_gangs<Config>::value
                            && !acc::has::num_workers<Config>::value
                            && !acc::has::num_vectors<Config>::value>::type
    forall(const acc_parallel_exec<InnerPolicy, Config>&,
           Iterable&& iter,
           Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel
  {
    forall<InnerPolicy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<!acc::has::num_gangs<Config>::value
                            && !acc::has::num_workers<Config>::value
                            && !acc::has::num_vectors<Config>::value>::type
    forall_Icount(const acc_parallel_exec<InnerPolicy, Config>&,
                  Iterable&& iter,
                  Index_type icount,
                  Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel
  {
    forall_Icount<InnerPolicy>(std::forward<Iterable>(iter),
                               icount,
                               std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<acc::has::num_gangs<Config>::value
                            && !acc::has::num_workers<Config>::value
                            && !acc::has::num_vectors<Config>::value>::type
    forall(const acc_parallel_exec<InnerPolicy, Config>&,
           Iterable&& iter,
           Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel num_gangs(Config::num_gangs)
  {
    forall<InnerPolicy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<acc::has::num_gangs<Config>::value
                            && !acc::has::num_workers<Config>::value
                            && !acc::has::num_vectors<Config>::value>::type
    forall_Icount(const acc_parallel_exec<InnerPolicy, Config>&,
                  Iterable&& iter,
                  Index_type icount,
                  Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel num_gangs(Config::num_gangs)
  {
    forall_Icount<InnerPolicy>(std::forward<Iterable>(iter),
                               icount,
                               std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<!acc::has::num_gangs<Config>::value
                            && acc::has::num_workers<Config>::value
                            && !acc::has::num_vectors<Config>::value>::type
    forall(const acc_parallel_exec<InnerPolicy, Config>&,
           Iterable&& iter,
           Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel num_workers(Config::num_workers)
  {
    forall<InnerPolicy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<!acc::has::num_gangs<Config>::value
                            && acc::has::num_workers<Config>::value
                            && !acc::has::num_vectors<Config>::value>::type
    forall_Icount(const acc_parallel_exec<InnerPolicy, Config>&,
                  Iterable&& iter,
                  Index_type icount,
                  Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel num_workers(Config::num_workers)
  {
    forall_Icount<InnerPolicy>(std::forward<Iterable>(iter),
                               icount,
                               std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<!acc::has::num_gangs<Config>::value
                            && !acc::has::num_workers<Config>::value
                            && acc::has::num_vectors<Config>::value>::type
    forall(const acc_parallel_exec<InnerPolicy, Config>&,
           Iterable&& iter,
           Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel vector_length(Config::num_vectors)
  {
    forall<InnerPolicy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<!acc::has::num_gangs<Config>::value
                            && !acc::has::num_workers<Config>::value
                            && acc::has::num_vectors<Config>::value>::type
    forall_Icount(const acc_parallel_exec<InnerPolicy, Config>&,
                  Iterable&& iter,
                  Index_type icount,
                  Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel vector_length(Config::num_vectors)
  {
    forall_Icount<InnerPolicy>(std::forward<Iterable>(iter),
                               icount,
                               std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<acc::has::num_gangs<Config>::value
                            && acc::has::num_workers<Config>::value
                            && !acc::has::num_vectors<Config>::value>::type
    forall(const acc_parallel_exec<InnerPolicy, Config>&,
           Iterable&& iter,
           Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel num_gangs(Config::num_gangs) \
    num_workers(Config::num_workers)
  {
    forall<InnerPolicy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<acc::has::num_gangs<Config>::value
                            && acc::has::num_workers<Config>::value
                            && !acc::has::num_vectors<Config>::value>::type
    forall_Icount(const acc_parallel_exec<InnerPolicy, Config>&,
                  Iterable&& iter,
                  Index_type icount,
                  Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel num_gangs(Config::num_gangs) \
    num_workers(Config::num_workers)
  {
    forall_Icount<InnerPolicy>(std::forward<Iterable>(iter),
                               icount,
                               std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<!acc::has::num_gangs<Config>::value
                            && acc::has::num_workers<Config>::value
                            && acc::has::num_vectors<Config>::value>::type
    forall(const acc_parallel_exec<InnerPolicy, Config>&,
           Iterable&& iter,
           Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel num_workers(Config::num_workers) \
    vector_length(Config::num_vectors)
  {
    forall<InnerPolicy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<!acc::has::num_gangs<Config>::value
                            && acc::has::num_workers<Config>::value
                            && acc::has::num_vectors<Config>::value>::type
    forall_Icount(const acc_parallel_exec<InnerPolicy, Config>&,
                  Iterable&& iter,
                  Index_type icount,
                  Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel num_workers(Config::num_workers) \
    vector_length(Config::num_vectors)
  {
    forall_Icount<InnerPolicy>(std::forward<Iterable>(iter),
                               icount,
                               std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<acc::has::num_gangs<Config>::value
                            && !acc::has::num_workers<Config>::value
                            && acc::has::num_vectors<Config>::value>::type
    forall(const acc_parallel_exec<InnerPolicy, Config>&,
           Iterable&& iter,
           Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel num_gangs(Config::num_gangs) \
    vector_length(Config::num_vectors)
  {
    forall<InnerPolicy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<acc::has::num_gangs<Config>::value
                            && !acc::has::num_workers<Config>::value
                            && acc::has::num_vectors<Config>::value>::type
    forall_Icount(const acc_parallel_exec<InnerPolicy, Config>&,
                  Iterable&& iter,
                  Index_type icount,
                  Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel num_gangs(Config::num_gangs) \
    vector_length(Config::num_vectors)
  {
    forall_Icount<InnerPolicy>(std::forward<Iterable>(iter),
                               icount,
                               std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<acc::has::num_gangs<Config>::value
                            && acc::has::num_workers<Config>::value
                            && acc::has::num_vectors<Config>::value>::type
    forall(const acc_parallel_exec<InnerPolicy, Config>&,
           Iterable&& iter,
           Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel num_gangs(Config::num_gangs) \
    num_workers(Config::num_workers) vector_length(Config::num_vectors)
  {
    forall<InnerPolicy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<acc::has::num_gangs<Config>::value
                            && acc::has::num_workers<Config>::value
                            && acc::has::num_vectors<Config>::value>::type
    forall_Icount(const acc_parallel_exec<InnerPolicy, Config>&,
                  Iterable&& iter,
                  Index_type icount,
                  Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel num_gangs(Config::num_gangs) \
    num_workers(Config::num_workers) vector_length(Config::num_vectors)
  {
    forall_Icount<InnerPolicy>(std::forward<Iterable>(iter),
                               icount,
                               std::forward<Func>(body));
  }
}


template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<!acc::has::num_gangs<Config>::value
                            && !acc::has::num_workers<Config>::value
                            && !acc::has::num_vectors<Config>::value>::type
    forall(const acc_kernels_exec<InnerPolicy, Config>&,
           Iterable&& iter,
           Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels
  {
    forall<InnerPolicy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<!acc::has::num_gangs<Config>::value
                            && !acc::has::num_workers<Config>::value
                            && !acc::has::num_vectors<Config>::value>::type
    forall_Icount(const acc_kernels_exec<InnerPolicy, Config>&,
                  Iterable&& iter,
                  Index_type icount,
                  Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels
  {
    forall_Icount<InnerPolicy>(std::forward<Iterable>(iter),
                               icount,
                               std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<acc::has::num_gangs<Config>::value
                            && !acc::has::num_workers<Config>::value
                            && !acc::has::num_vectors<Config>::value>::type
    forall(const acc_kernels_exec<InnerPolicy, Config>&,
           Iterable&& iter,
           Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels num_gangs(Config::num_gangs)
  {
    forall<InnerPolicy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<acc::has::num_gangs<Config>::value
                            && !acc::has::num_workers<Config>::value
                            && !acc::has::num_vectors<Config>::value>::type
    forall_Icount(const acc_kernels_exec<InnerPolicy, Config>&,
                  Iterable&& iter,
                  Index_type icount,
                  Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels num_gangs(Config::num_gangs)
  {
    forall_Icount<InnerPolicy>(std::forward<Iterable>(iter),
                               icount,
                               std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<!acc::has::num_gangs<Config>::value
                            && acc::has::num_workers<Config>::value
                            && !acc::has::num_vectors<Config>::value>::type
    forall(const acc_kernels_exec<InnerPolicy, Config>&,
           Iterable&& iter,
           Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels num_workers(Config::num_workers)
  {
    forall<InnerPolicy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<!acc::has::num_gangs<Config>::value
                            && acc::has::num_workers<Config>::value
                            && !acc::has::num_vectors<Config>::value>::type
    forall_Icount(const acc_kernels_exec<InnerPolicy, Config>&,
                  Iterable&& iter,
                  Index_type icount,
                  Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels num_workers(Config::num_workers)
  {
    forall_Icount<InnerPolicy>(std::forward<Iterable>(iter),
                               icount,
                               std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<!acc::has::num_gangs<Config>::value
                            && !acc::has::num_workers<Config>::value
                            && acc::has::num_vectors<Config>::value>::type
    forall(const acc_kernels_exec<InnerPolicy, Config>&,
           Iterable&& iter,
           Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels vector_length(Config::num_vectors)
  {
    forall<InnerPolicy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<!acc::has::num_gangs<Config>::value
                            && !acc::has::num_workers<Config>::value
                            && acc::has::num_vectors<Config>::value>::type
    forall_Icount(const acc_kernels_exec<InnerPolicy, Config>&,
                  Iterable&& iter,
                  Index_type icount,
                  Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels vector_length(Config::num_vectors)
  {
    forall_Icount<InnerPolicy>(std::forward<Iterable>(iter),
                               icount,
                               std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<acc::has::num_gangs<Config>::value
                            && acc::has::num_workers<Config>::value
                            && !acc::has::num_vectors<Config>::value>::type
    forall(const acc_kernels_exec<InnerPolicy, Config>&,
           Iterable&& iter,
           Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels num_gangs(Config::num_gangs) \
    num_workers(Config::num_workers)
  {
    forall<InnerPolicy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<acc::has::num_gangs<Config>::value
                            && acc::has::num_workers<Config>::value
                            && !acc::has::num_vectors<Config>::value>::type
    forall_Icount(const acc_kernels_exec<InnerPolicy, Config>&,
                  Iterable&& iter,
                  Index_type icount,
                  Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels num_gangs(Config::num_gangs) \
    num_workers(Config::num_workers)
  {
    forall_Icount<InnerPolicy>(std::forward<Iterable>(iter),
                               icount,
                               std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<!acc::has::num_gangs<Config>::value
                            && acc::has::num_workers<Config>::value
                            && acc::has::num_vectors<Config>::value>::type
    forall(const acc_kernels_exec<InnerPolicy, Config>&,
           Iterable&& iter,
           Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels num_workers(Config::num_workers) \
    vector_length(Config::num_vectors)
  {
    forall<InnerPolicy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<!acc::has::num_gangs<Config>::value
                            && acc::has::num_workers<Config>::value
                            && acc::has::num_vectors<Config>::value>::type
    forall_Icount(const acc_kernels_exec<InnerPolicy, Config>&,
                  Iterable&& iter,
                  Index_type icount,
                  Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels num_workers(Config::num_workers) \
    vector_length(Config::num_vectors)
  {
    forall_Icount<InnerPolicy>(std::forward<Iterable>(iter),
                               icount,
                               std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<acc::has::num_gangs<Config>::value
                            && !acc::has::num_workers<Config>::value
                            && acc::has::num_vectors<Config>::value>::type
    forall(const acc_kernels_exec<InnerPolicy, Config>&,
           Iterable&& iter,
           Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels num_gangs(Config::num_gangs) \
    vector_length(Config::num_vectors)
  {
    forall<InnerPolicy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<acc::has::num_gangs<Config>::value
                            && !acc::has::num_workers<Config>::value
                            && acc::has::num_vectors<Config>::value>::type
    forall_Icount(const acc_kernels_exec<InnerPolicy, Config>&,
                  Iterable&& iter,
                  Index_type icount,
                  Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels num_gangs(Config::num_gangs) \
    vector_length(Config::num_vectors)
  {
    forall_Icount<InnerPolicy>(std::forward<Iterable>(iter),
                               icount,
                               std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<acc::has::num_gangs<Config>::value
                            && acc::has::num_workers<Config>::value
                            && acc::has::num_vectors<Config>::value>::type
    forall(const acc_kernels_exec<InnerPolicy, Config>&,
           Iterable&& iter,
           Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels num_gangs(Config::num_gangs) \
    num_workers(Config::num_workers) vector_length(Config::num_vectors)
  {
    forall<InnerPolicy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable,
          typename InnerPolicy,
          typename Config,
          typename Func>
RAJA_INLINE
    typename std::enable_if<acc::has::num_gangs<Config>::value
                            && acc::has::num_workers<Config>::value
                            && acc::has::num_vectors<Config>::value>::type
    forall_Icount(const acc_kernels_exec<InnerPolicy, Config>&,
                  Iterable&& iter,
                  Index_type icount,
                  Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels num_gangs(Config::num_gangs) \
    num_workers(Config::num_workers) vector_length(Config::num_vectors)
  {
    forall_Icount<InnerPolicy>(std::forward<Iterable>(iter),
                               icount,
                               std::forward<Func>(body));
  }
}

///
/// OpenACC parallel for policy implementation
///

template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<!acc::has::is_independent<Config>::value
                           && !acc::has::is_gang<Config>::value
                           && !acc::has::is_worker<Config>::value
                           && !acc::has::is_vector<Config>::value>::type
forall(const acc_loop_exec<Config>&, Iterable&& iter, Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(begin[i]);
  }
}

template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<!acc::has::is_independent<Config>::value
                           && !acc::has::is_gang<Config>::value
                           && !acc::has::is_worker<Config>::value
                           && !acc::has::is_vector<Config>::value>::type
forall_Icount(const acc_loop_exec<Config>&,
              Iterable&& iter,
              Index_type icount,
              Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(i + icount, begin[i]);
  }
}

template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<!acc::has::is_independent<Config>::value
                           && acc::has::is_gang<Config>::value
                           && !acc::has::is_worker<Config>::value
                           && !acc::has::is_vector<Config>::value>::type
forall(const acc_loop_exec<Config>&, Iterable&& iter, Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop gang
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(begin[i]);
  }
}

template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<!acc::has::is_independent<Config>::value
                           && acc::has::is_gang<Config>::value
                           && !acc::has::is_worker<Config>::value
                           && !acc::has::is_vector<Config>::value>::type
forall_Icount(const acc_loop_exec<Config>&,
              Iterable&& iter,
              Index_type icount,
              Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop gang
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(i + icount, begin[i]);
  }
}
template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<!acc::has::is_independent<Config>::value
                           && !acc::has::is_gang<Config>::value
                           && acc::has::is_worker<Config>::value
                           && !acc::has::is_vector<Config>::value>::type
forall(const acc_loop_exec<Config>&, Iterable&& iter, Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop worker
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(begin[i]);
  }
}

template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<!acc::has::is_independent<Config>::value
                           && !acc::has::is_gang<Config>::value
                           && acc::has::is_worker<Config>::value
                           && !acc::has::is_vector<Config>::value>::type
forall_Icount(const acc_loop_exec<Config>&,
              Iterable&& iter,
              Index_type icount,
              Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop worker
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(i + icount, begin[i]);
  }
}
template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<!acc::has::is_independent<Config>::value
                           && !acc::has::is_gang<Config>::value
                           && !acc::has::is_worker<Config>::value
                           && acc::has::is_vector<Config>::value>::type
forall(const acc_loop_exec<Config>&, Iterable&& iter, Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop vector
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(begin[i]);
  }
}

template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<!acc::has::is_independent<Config>::value
                           && !acc::has::is_gang<Config>::value
                           && !acc::has::is_worker<Config>::value
                           && acc::has::is_vector<Config>::value>::type
forall_Icount(const acc_loop_exec<Config>&,
              Iterable&& iter,
              Index_type icount,
              Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop vector
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(i + icount, begin[i]);
  }
}

template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<!acc::has::is_independent<Config>::value
                           && acc::has::is_gang<Config>::value
                           && !acc::has::is_worker<Config>::value
                           && acc::has::is_vector<Config>::value>::type
forall(const acc_loop_exec<Config>&, Iterable&& iter, Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop gang vector
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(begin[i]);
  }
}

template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<!acc::has::is_independent<Config>::value
                           && acc::has::is_gang<Config>::value
                           && !acc::has::is_worker<Config>::value
                           && acc::has::is_vector<Config>::value>::type
forall_Icount(const acc_loop_exec<Config>&,
              Iterable&& iter,
              Index_type icount,
              Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop gang vector
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(i + icount, begin[i]);
  }
}

template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<!acc::has::is_independent<Config>::value
                           && acc::has::is_gang<Config>::value
                           && acc::has::is_worker<Config>::value
                           && !acc::has::is_vector<Config>::value>::type
forall(const acc_loop_exec<Config>&, Iterable&& iter, Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop gang worker
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(begin[i]);
  }
}

template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<!acc::has::is_independent<Config>::value
                           && acc::has::is_gang<Config>::value
                           && acc::has::is_worker<Config>::value
                           && !acc::has::is_vector<Config>::value>::type
forall_Icount(const acc_loop_exec<Config>&,
              Iterable&& iter,
              Index_type icount,
              Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop gang worker
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(i + icount, begin[i]);
  }
}

template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<!acc::has::is_independent<Config>::value
                           && !acc::has::is_gang<Config>::value
                           && acc::has::is_worker<Config>::value
                           && acc::has::is_vector<Config>::value>::type
forall(const acc_loop_exec<Config>&, Iterable&& iter, Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop worker vector
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(begin[i]);
  }
}

template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<!acc::has::is_independent<Config>::value
                           && !acc::has::is_gang<Config>::value
                           && acc::has::is_worker<Config>::value
                           && acc::has::is_vector<Config>::value>::type
forall_Icount(const acc_loop_exec<Config>&,
              Iterable&& iter,
              Index_type icount,
              Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop worker vector
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(i + icount, begin[i]);
  }
}

template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<!acc::has::is_independent<Config>::value
                           && acc::has::is_gang<Config>::value
                           && acc::has::is_worker<Config>::value
                           && acc::has::is_vector<Config>::value>::type
forall(const acc_loop_exec<Config>&, Iterable&& iter, Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop gang worker vector
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(begin[i]);
  }
}

template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<!acc::has::is_independent<Config>::value
                           && acc::has::is_gang<Config>::value
                           && acc::has::is_worker<Config>::value
                           && acc::has::is_vector<Config>::value>::type
forall_Icount(const acc_loop_exec<Config>&,
              Iterable&& iter,
              Index_type icount,
              Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop gang worker vector
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(i + icount, begin[i]);
  }
}


template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<acc::has::is_independent<Config>::value
                           && !acc::has::is_gang<Config>::value
                           && !acc::has::is_worker<Config>::value
                           && !acc::has::is_vector<Config>::value>::type
forall(const acc_loop_exec<Config>&, Iterable&& iter, Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop independent
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(begin[i]);
  }
}

template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<acc::has::is_independent<Config>::value
                           && !acc::has::is_gang<Config>::value
                           && !acc::has::is_worker<Config>::value
                           && !acc::has::is_vector<Config>::value>::type
forall_Icount(const acc_loop_exec<Config>&,
              Iterable&& iter,
              Index_type icount,
              Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop independent
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(i + icount, begin[i]);
  }
}

template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<acc::has::is_independent<Config>::value
                           && acc::has::is_gang<Config>::value
                           && !acc::has::is_worker<Config>::value
                           && !acc::has::is_vector<Config>::value>::type
forall(const acc_loop_exec<Config>&, Iterable&& iter, Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop independent gang
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(begin[i]);
  }
}

template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<acc::has::is_independent<Config>::value
                           && acc::has::is_gang<Config>::value
                           && !acc::has::is_worker<Config>::value
                           && !acc::has::is_vector<Config>::value>::type
forall_Icount(const acc_loop_exec<Config>&,
              Iterable&& iter,
              Index_type icount,
              Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop independent gang
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(i + icount, begin[i]);
  }
}
template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<acc::has::is_independent<Config>::value
                           && !acc::has::is_gang<Config>::value
                           && acc::has::is_worker<Config>::value
                           && !acc::has::is_vector<Config>::value>::type
forall(const acc_loop_exec<Config>&, Iterable&& iter, Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop independent worker
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(begin[i]);
  }
}

template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<acc::has::is_independent<Config>::value
                           && !acc::has::is_gang<Config>::value
                           && acc::has::is_worker<Config>::value
                           && !acc::has::is_vector<Config>::value>::type
forall_Icount(const acc_loop_exec<Config>&,
              Iterable&& iter,
              Index_type icount,
              Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop independent worker
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(i + icount, begin[i]);
  }
}
template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<acc::has::is_independent<Config>::value
                           && !acc::has::is_gang<Config>::value
                           && !acc::has::is_worker<Config>::value
                           && acc::has::is_vector<Config>::value>::type
forall(const acc_loop_exec<Config>&, Iterable&& iter, Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop independent vector
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(begin[i]);
  }
}

template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<acc::has::is_independent<Config>::value
                           && !acc::has::is_gang<Config>::value
                           && !acc::has::is_worker<Config>::value
                           && acc::has::is_vector<Config>::value>::type
forall_Icount(const acc_loop_exec<Config>&,
              Iterable&& iter,
              Index_type icount,
              Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop independent vector
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(i + icount, begin[i]);
  }
}

template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<acc::has::is_independent<Config>::value
                           && acc::has::is_gang<Config>::value
                           && !acc::has::is_worker<Config>::value
                           && acc::has::is_vector<Config>::value>::type
forall(const acc_loop_exec<Config>&, Iterable&& iter, Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop independent gang vector
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(begin[i]);
  }
}

template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<acc::has::is_independent<Config>::value
                           && acc::has::is_gang<Config>::value
                           && !acc::has::is_worker<Config>::value
                           && acc::has::is_vector<Config>::value>::type
forall_Icount(const acc_loop_exec<Config>&,
              Iterable&& iter,
              Index_type icount,
              Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop independent gang vector
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(i + icount, begin[i]);
  }
}

template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<acc::has::is_independent<Config>::value
                           && acc::has::is_gang<Config>::value
                           && acc::has::is_worker<Config>::value
                           && !acc::has::is_vector<Config>::value>::type
forall(const acc_loop_exec<Config>&, Iterable&& iter, Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop independent gang worker
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(begin[i]);
  }
}

template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<acc::has::is_independent<Config>::value
                           && acc::has::is_gang<Config>::value
                           && acc::has::is_worker<Config>::value
                           && !acc::has::is_vector<Config>::value>::type
forall_Icount(const acc_loop_exec<Config>&,
              Iterable&& iter,
              Index_type icount,
              Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop independent gang worker
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(i + icount, begin[i]);
  }
}

template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<acc::has::is_independent<Config>::value
                           && !acc::has::is_gang<Config>::value
                           && acc::has::is_worker<Config>::value
                           && acc::has::is_vector<Config>::value>::type
forall(const acc_loop_exec<Config>&, Iterable&& iter, Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop independent worker vector
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(begin[i]);
  }
}

template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<acc::has::is_independent<Config>::value
                           && !acc::has::is_gang<Config>::value
                           && acc::has::is_worker<Config>::value
                           && acc::has::is_vector<Config>::value>::type
forall_Icount(const acc_loop_exec<Config>&,
              Iterable&& iter,
              Index_type icount,
              Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop independent worker vector
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(i + icount, begin[i]);
  }
}

template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<acc::has::is_independent<Config>::value
                           && acc::has::is_gang<Config>::value
                           && acc::has::is_worker<Config>::value
                           && acc::has::is_vector<Config>::value>::type
forall(const acc_loop_exec<Config>&, Iterable&& iter, Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop independent gang worker vector
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(begin[i]);
  }
}

template <typename Config, typename Iterable, typename Func>
RAJA_INLINE typename std::enable_if<acc::has::is_independent<Config>::value
                           && acc::has::is_gang<Config>::value
                           && acc::has::is_worker<Config>::value
                           && acc::has::is_vector<Config>::value>::type
forall_Icount(const acc_loop_exec<Config>&,
              Iterable&& iter,
              Index_type icount,
              Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop independent gang worker vector
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(i + icount, begin[i]);
  }
}

}  // closing brace for impl namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for if defined(RAJA_ENABLE_OPENACC)

#endif  // closing endif for header file include guard
