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

#include "RAJA/index/RangeSegment.hpp"

#include "RAJA/policy/openacc/policy.hpp"

namespace RAJA
{

namespace impl
{
using namespace RAJA::acc;

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc parallel")
RAJA_INLINE When<Config, no_ngangs, no_nworkers, no_nvectors> forall(
    const acc_parallel_exec<Policy, Config>&,
    Iterable&& iter,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc parallel")
RAJA_INLINE When<Config, no_ngangs, no_nworkers, no_nvectors> forall_Icount(
    const acc_parallel_exec<Policy, Config>&,
    Iterable&& iter,
    Index_type icount,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc parallel ngangs")
RAJA_INLINE When<Config, ngangs, no_nworkers, no_nvectors> forall(
    const acc_parallel_exec<Policy, Config>&,
    Iterable&& iter,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel num_gangs(Config::ngangs)
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc parallel ngangs")
RAJA_INLINE When<Config, ngangs, no_nworkers, no_nvectors> forall_Icount(
    const acc_parallel_exec<Policy, Config>&,
    Iterable&& iter,
    Index_type icount,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel num_gangs(Config::ngangs)
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc parallel nworkers")
RAJA_INLINE When<Config, no_ngangs, nworkers, no_nvectors> forall(
    const acc_parallel_exec<Policy, Config>&,
    Iterable&& iter,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel num_workers(Config::nworkers)
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc parallel nworkers")
RAJA_INLINE When<Config, no_ngangs, nworkers, no_nvectors> forall_Icount(
    const acc_parallel_exec<Policy, Config>&,
    Iterable&& iter,
    Index_type icount,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel num_workers(Config::nworkers)
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc parallel nvectors")
RAJA_INLINE When<Config, no_ngangs, no_nworkers, nvectors> forall(
    const acc_parallel_exec<Policy, Config>&,
    Iterable&& iter,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel vector_length(nvectors)
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc parallel nvectors")
RAJA_INLINE When<Config, no_ngangs, no_nworkers, nvectors> forall_Icount(
    const acc_parallel_exec<Policy, Config>&,
    Iterable&& iter,
    Index_type icount,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel vector_length(nvectors)
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc parallel ngangs nworkers")
RAJA_INLINE When<Config, ngangs, nworkers, no_nvectors> forall(
    const acc_parallel_exec<Policy, Config>&,
    Iterable&& iter,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel num_gangs(Config::ngangs) num_workers(Config::nworkers)
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc parallel ngangs nworkers")
RAJA_INLINE When<Config, ngangs, nworkers, no_nvectors> forall_Icount(
    const acc_parallel_exec<Policy, Config>&,
    Iterable&& iter,
    Index_type icount,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel num_gangs(Config::ngangs) num_workers(Config::nworkers)
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc parallel ngangs nvectors")
RAJA_INLINE When<Config, ngangs, no_nworkers, nvectors> forall(
    const acc_parallel_exec<Policy, Config>&,
    Iterable&& iter,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel num_gangs(Config::ngangs) vector_length(nvectors >
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc parallel ngangs nvectors")
RAJA_INLINE When<Config, ngangs, no_nworkers, nvectors> forall_Icount(
    const acc_parallel_exec<Policy, Config>&,
    Iterable&& iter,
    Index_type icount,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel num_gangs(Config::ngangs) vector_length(nvectors)
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc parallel nworkers nvectors")
RAJA_INLINE When<Config, no_ngangs, nworkers, nvectors> forall(
    const acc_parallel_exec<Policy, Config>&,
    Iterable&& iter,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel num_workers(Config::nworkers) vector_length(nvectors)
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc parallel nworkers nvectors")
RAJA_INLINE When<Config, no_ngangs, nworkers, nvectors> forall_Icount(
    const acc_parallel_exec<Policy, Config>&,
    Iterable&& iter,
    Index_type icount,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel num_workers(Config::nworkers) vector_length(nvectors)
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc parallel ngangs nworkers nvectors")
RAJA_INLINE When<Config, ngangs, nworkers, nvectors> forall(
    const acc_parallel_exec<Policy, Config>&,
    Iterable&& iter,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel num_gangs(Config::ngangs) num_workers(Config::nworkers) \
    vector_length(nvectors)
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc parallel ngangs nworkers nvectors")
RAJA_INLINE When<Config, ngangs, nworkers, nvectors> forall_Icount(
    const acc_parallel_exec<Policy, Config>&,
    Iterable&& iter,
    Index_type icount,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc parallel num_gangs(Config::ngangs) num_workers(Config::nworkers) \
    vector_length(nvectors)
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}


template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc kernels")
RAJA_INLINE When<Config, no_ngangs, no_nworkers, no_nvectors> forall(
    const acc_kernels_exec<Policy, Config>&,
    Iterable&& iter,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc kernels")
RAJA_INLINE When<Config, no_ngangs, no_nworkers, no_nvectors> forall_Icount(
    const acc_kernels_exec<Policy, Config>&,
    Iterable&& iter,
    Index_type icount,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc kernels ngangs")
RAJA_INLINE When<Config, ngangs, no_nworkers, no_nvectors> forall(
    const acc_kernels_exec<Policy, Config>&,
    Iterable&& iter,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels num_gangs(Config::ngangs)
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc kernels ngangs")
RAJA_INLINE When<Config, ngangs, no_nworkers, no_nvectors> forall_Icount(
    const acc_kernels_exec<Policy, Config>&,
    Iterable&& iter,
    Index_type icount,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels num_gangs(Config::ngangs)
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc kernels nworkers")
RAJA_INLINE When<Config, no_ngangs, nworkers, no_nvectors> forall(
    const acc_kernels_exec<Policy, Config>&,
    Iterable&& iter,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels num_workers(Config::nworkers)
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc kernels nworkers")
RAJA_INLINE When<Config, no_ngangs, nworkers, no_nvectors> forall_Icount(
    const acc_kernels_exec<Policy, Config>&,
    Iterable&& iter,
    Index_type icount,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels num_workers(Config::nworkers)
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc kernels nvectors")
RAJA_INLINE When<Config, no_ngangs, no_nworkers, nvectors> forall(
    const acc_kernels_exec<Policy, Config>&,
    Iterable&& iter,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels vector_length(nvectors)
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc kernels nvectors")
RAJA_INLINE When<Config, no_ngangs, no_nworkers, nvectors> forall_Icount(
    const acc_kernels_exec<Policy, Config>&,
    Iterable&& iter,
    Index_type icount,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels vector_length(nvectors)
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc kernels ngangs nworkers")
RAJA_INLINE When<Config, ngangs, nworkers, no_nvectors> forall(
    const acc_kernels_exec<Policy, Config>&,
    Iterable&& iter,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels num_gangs(Config::ngangs) num_workers(Config::nworkers)
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc kernels ngangs nworkers")
RAJA_INLINE When<Config, ngangs, nworkers, no_nvectors> forall_Icount(
    const acc_kernels_exec<Policy, Config>&,
    Iterable&& iter,
    Index_type icount,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels num_gangs(Config::ngangs) num_workers(Config::nworkers)
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc kernels ngangs nvectors")
RAJA_INLINE When<Config, ngangs, no_nworkers, nvectors> forall(
    const acc_kernels_exec<Policy, Config>&,
    Iterable&& iter,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels num_gangs(Config::ngangs) vector_length(nvectors >
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc kernels ngangs nvectors")
RAJA_INLINE When<Config, ngangs, no_nworkers, nvectors> forall_Icount(
    const acc_kernels_exec<Policy, Config>&,
    Iterable&& iter,
    Index_type icount,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels num_gangs(Config::ngangs) vector_length(nvectors)
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc kernels nworkers nvectors")
RAJA_INLINE When<Config, no_ngangs, nworkers, nvectors> forall(
    const acc_kernels_exec<Policy, Config>&,
    Iterable&& iter,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels num_workers(Config::nworkers) vector_length(nvectors)
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc kernels nworkers nvectors")
RAJA_INLINE When<Config, no_ngangs, nworkers, nvectors> forall_Icount(
    const acc_kernels_exec<Policy, Config>&,
    Iterable&& iter,
    Index_type icount,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels num_workers(Config::nworkers) vector_length(nvectors)
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc kernels ngangs nworkers nvectors")
RAJA_INLINE When<Config, ngangs, nworkers, nvectors> forall(
    const acc_kernels_exec<Policy, Config>&,
    Iterable&& iter,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels num_gangs(Config::ngangs) num_workers(Config::nworkers) \
    vector_length(nvectors)
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

template <typename Iterable, typename Policy, typename Config, typename Func>
RAJA_VERBOSE("\nacc kernels ngangs nworkers nvectors")
RAJA_INLINE When<Config, ngangs, nworkers, nvectors> forall_Icount(
    const acc_kernels_exec<Policy, Config>&,
    Iterable&& iter,
    Index_type icount,
    Func&& loop_body)
{
  using body_type = typename std::remove_reference<decltype(loop_body)>::type;
  body_type body = loop_body;
#pragma acc kernels num_gangs(Config::ngangs) num_workers(Config::nworkers) \
    vector_length(nvectors)
  {
    forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body));
  }
}

///
/// OpenACC loop policy implementation
///

template <typename Config, typename Iterable, typename Func>
RAJA_VERBOSE("\nacc loop")
RAJA_INLINE When<Config, no_independent, no_gang, no_worker, no_vector> forall(
    const acc_loop_exec<Config>&,
    Iterable&& iter,
    Func&& loop_body)
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
RAJA_VERBOSE("\nacc loop vector")
RAJA_INLINE When<Config, no_independent, no_gang, no_worker, vector> forall(
    const acc_loop_exec<Config>&,
    Iterable&& iter,
    Func&& loop_body)
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
RAJA_VERBOSE("\nacc loop worker")
RAJA_INLINE When<Config, no_independent, no_gang, worker, no_vector> forall(
    const acc_loop_exec<Config>&,
    Iterable&& iter,
    Func&& loop_body)
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
RAJA_VERBOSE("\nacc loop worker vector")
RAJA_INLINE When<Config, no_independent, no_gang, worker, vector> forall(
    const acc_loop_exec<Config>&,
    Iterable&& iter,
    Func&& loop_body)
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
RAJA_VERBOSE("\nacc loop gang")
RAJA_INLINE When<Config, no_independent, gang, no_worker, no_vector> forall(
    const acc_loop_exec<Config>&,
    Iterable&& iter,
    Func&& loop_body)
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
RAJA_VERBOSE("\nacc loop gang vector")
RAJA_INLINE When<Config, no_independent, gang, no_worker, vector> forall(
    const acc_loop_exec<Config>&,
    Iterable&& iter,
    Func&& loop_body)
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
RAJA_VERBOSE("\nacc loop gang worker")
RAJA_INLINE When<Config, no_independent, gang, worker, no_vector> forall(
    const acc_loop_exec<Config>&,
    Iterable&& iter,
    Func&& loop_body)
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
RAJA_VERBOSE("\nacc loop gang worker vector")
RAJA_INLINE When<Config, no_independent, gang, worker, vector> forall(
    const acc_loop_exec<Config>&,
    Iterable&& iter,
    Func&& loop_body)
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
RAJA_VERBOSE("\nacc loop independent")
RAJA_INLINE When<Config, independent, no_gang, no_worker, no_vector> forall(
    const acc_loop_exec<Config>&,
    Iterable&& iter,
    Func&& loop_body)
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
RAJA_VERBOSE("\nacc loop independent vector")
RAJA_INLINE When<Config, independent, no_gang, no_worker, vector> forall(
    const acc_loop_exec<Config>&,
    Iterable&& iter,
    Func&& loop_body)
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
RAJA_VERBOSE("\nacc loop independent worker")
RAJA_INLINE When<Config, independent, no_gang, worker, no_vector> forall(
    const acc_loop_exec<Config>&,
    Iterable&& iter,
    Func&& loop_body)
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
RAJA_VERBOSE("\nacc loop independent worker vector")
RAJA_INLINE When<Config, independent, no_gang, worker, vector> forall(
    const acc_loop_exec<Config>&,
    Iterable&& iter,
    Func&& loop_body)
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
RAJA_VERBOSE("\nacc loop independent gang")
RAJA_INLINE When<Config, independent, gang, no_worker, no_vector> forall(
    const acc_loop_exec<Config>&,
    Iterable&& iter,
    Func&& loop_body)
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
RAJA_VERBOSE("\nacc loop independent gang vector")
RAJA_INLINE When<Config, independent, gang, no_worker, vector> forall(
    const acc_loop_exec<Config>&,
    Iterable&& iter,
    Func&& loop_body)
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
RAJA_VERBOSE("\nacc loop independent gang worker")
RAJA_INLINE When<Config, independent, gang, worker, no_vector> forall(
    const acc_loop_exec<Config>&,
    Iterable&& iter,
    Func&& loop_body)
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
RAJA_VERBOSE("\nacc loop independent gang worker vector")
RAJA_INLINE When<Config, independent, gang, worker, vector> forall(
    const acc_loop_exec<Config>&,
    Iterable&& iter,
    Func&& loop_body)
{
  auto begin = std::begin(iter);
  auto end = std::end(iter);
  auto distance = std::distance(begin, end);
#pragma acc loop independent gang worker vector
  for (Index_type i = 0; i < distance; ++i) {
    loop_body(begin[i]);
  }
}

}  // closing brace for impl namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for if defined(RAJA_ENABLE_OPENACC)

#endif  // closing endif for header file include guard
