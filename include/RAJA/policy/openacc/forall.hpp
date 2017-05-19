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
#define STR_(X) #X
#define STR(X) STR_(X)

#define PRAGMA_ACC(exec, args) PRAGMA(acc exec args)
#define PRAGMA(x) _Pragma(#x)

#define FORALL(EXEC_TYPE, GANGS, WORKERS, VECTORS, DESCRIPTOR)                \
  template <typename Iterable,                                                \
            typename Policy,                                                  \
            typename Config,                                                  \
            typename Func>                                                    \
  RAJA_VERBOSE("\nacc " STR(EXEC_TYPE) " " STR(DESCRIPTOR))                   \
  RAJA_INLINE acc::When<Config, acc::GANGS, acc::WORKERS, acc::VECTORS>       \
  forall(const acc_##EXEC_TYPE##_exec<Policy, Config>&,                       \
         Iterable&& iter,                                                     \
         Func&& loop_body)                                                    \
  {                                                                           \
    using body_type =                                                         \
        typename std::remove_reference<decltype(loop_body)>::type;            \
    body_type body = loop_body;                                               \
    PRAGMA_ACC(EXEC_TYPE, DESCRIPTOR)                                         \
    {                                                                         \
      forall<Policy>(std::forward<Iterable>(iter), std::forward<Func>(body)); \
    }                                                                         \
  }                                                                           \
                                                                              \
  template <typename Iterable,                                                \
            typename Policy,                                                  \
            typename Config,                                                  \
            typename Func>                                                    \
  RAJA_VERBOSE("\nacc " STR(EXEC_TYPE) " " STR(DESCRIPTOR))                   \
  RAJA_INLINE acc::When<Config, acc::GANGS, acc::WORKERS, acc::VECTORS>       \
  forall_Icount(const acc_##EXEC_TYPE##_exec<Policy, Config>&,                \
                Iterable&& iter,                                              \
                Index_type icount,                                            \
                Func&& loop_body)                                             \
  {                                                                           \
    using body_type =                                                         \
        typename std::remove_reference<decltype(loop_body)>::type;            \
    body_type body = loop_body;                                               \
    PRAGMA_ACC(EXEC_TYPE, DESCRIPTOR)                                         \
    {                                                                         \
      forall_Icount<Policy>(std::forward<Iterable>(iter),                     \
                            icount,                                           \
                            std::forward<Func>(body));                        \
    }                                                                         \
  }

FORALL(parallel, no_ngangs, no_nworkers, no_nvectors, )
FORALL(parallel, ngangs, no_nworkers, no_nvectors, num_gangs(Config::ngangs))
FORALL(parallel,
       no_ngangs,
       nworkers,
       no_nvectors,
       num_workers(Config::nworkers))
FORALL(parallel,
       ngangs,
       nworkers,
       no_nvectors,
       num_gangs(Config::ngangs) num_workers(Config::nworkers))
FORALL(parallel,
       no_ngangs,
       no_nworkers,
       nvectors,
       vector_length(Config::nvectors))
FORALL(parallel,
       ngangs,
       no_nworkers,
       nvectors,
       num_gangs(Config::ngangs) vector_length(Config::nvectors))
FORALL(parallel,
       no_ngangs,
       nworkers,
       nvectors,
       num_workers(Config::nworkers) vector_length(Config::nvectors))
FORALL(parallel,
       ngangs,
       nworkers,
       nvectors,
       num_gangs(Config::ngangs) num_workers(Config::nworkers)
           vector_length(Config::nvectors))
FORALL(kernels, no_ngangs, no_nworkers, no_nvectors, )
FORALL(kernels, ngangs, no_nworkers, no_nvectors, num_gangs(Config::ngangs))
FORALL(kernels, no_ngangs, nworkers, no_nvectors, num_workers(Config::nworkers))
FORALL(kernels,
       ngangs,
       nworkers,
       no_nvectors,
       num_gangs(Config::ngangs) num_workers(Config::nworkers))
FORALL(kernels,
       no_ngangs,
       no_nworkers,
       nvectors,
       vector_length(Config::nvectors))
FORALL(kernels,
       ngangs,
       no_nworkers,
       nvectors,
       num_gangs(Config::ngangs) vector_length(Config::nvectors))
FORALL(kernels,
       no_ngangs,
       nworkers,
       nvectors,
       num_workers(Config::nworkers) vector_length(Config::nvectors))
FORALL(kernels,
       ngangs,
       nworkers,
       nvectors,
       num_gangs(Config::ngangs) num_workers(Config::nworkers)
           vector_length(Config::nvectors))

#undef FORALL

///
/// OpenACC loop policy implementation
///

#define FORALL(IND, GANG, WORKER, VECTOR, DESCRIPTOR)                          \
  template <typename Config, typename Iterable, typename Func>                 \
  RAJA_VERBOSE("\nacc loop " STR(DESCRIPTOR))                                  \
  RAJA_INLINE acc::When<Config, acc::IND, acc::GANG, acc::WORKER, acc::VECTOR> \
  forall(const acc_loop_exec<Config>&, Iterable&& iter, Func&& loop_body)      \
  {                                                                            \
    auto begin = std::begin(iter);                                             \
    auto end = std::end(iter);                                                 \
    auto distance = std::distance(begin, end);                                 \
    PRAGMA_ACC(loop, DESCRIPTOR)                                               \
    for (Index_type i = 0; i < distance; ++i) {                                \
      loop_body(begin[i]);                                                     \
    }                                                                          \
  }

FORALL(no_independent, no_gang, no_worker, no_vector, )
FORALL(no_independent, no_gang, no_worker, vector, vector)
FORALL(no_independent, no_gang, worker, no_vector, worker)
FORALL(no_independent, no_gang, worker, vector, worker vector)
FORALL(no_independent, gang, no_worker, no_vector, gang)
FORALL(no_independent, gang, no_worker, vector, gang vector)
FORALL(no_independent, gang, worker, no_vector, gang worker)
FORALL(no_independent, gang, worker, vector, gang worker vector)
FORALL(independent, no_gang, no_worker, no_vector, independent)
FORALL(independent, no_gang, no_worker, vector, independent vector)
FORALL(independent, no_gang, worker, no_vector, independent worker)
FORALL(independent, no_gang, worker, vector, independent worker vector)
FORALL(independent, gang, no_worker, no_vector, independent gang)
FORALL(independent, gang, no_worker, vector, independent gang vector)
FORALL(independent, gang, worker, no_vector, independent gang worker)
FORALL(independent, gang, worker, vector, independent gang worker vector)

#undef FORALL

#undef STR_
#undef STR
#undef PRAGMA_ACC
#undef PRAGMA

}  // closing brace for impl namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for if defined(RAJA_ENABLE_OPENACC)

#endif  // closing endif for header file include guard
