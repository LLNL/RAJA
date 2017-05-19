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

struct ForallN_ACC_parallel_Tag {
};
template <typename Config = acc::config<>, typename NEXT = Execute>
struct ACC_Parallel : public Config {
  using PolicyTag = ForallN_ACC_parallel_Tag;
  using NextPolicy = NEXT;
};

struct ForallN_ACC_kernels_Tag {
};
template <typename Config = acc::config<>, typename NEXT = Execute>
struct ACC_Kernels : public Config {
  using PolicyTag = ForallN_ACC_kernels_Tag;
  using NextPolicy = NEXT;
};

/******************************************************************
 *  forallN_policy(), Openacc Parallel Region execution
 ******************************************************************/

#define STR_(X) #X
#define STR(X) STR_(X)

#define PRAGMA_ACC(exec, args) PRAGMA(acc exec args)
#define PRAGMA(x) _Pragma(#x)

#define FORALLN(EXEC_TYPE, GANGS, WORKERS, VECTORS, DESCRIPTOR)            \
  template <typename Policy, typename Body, typename... PArgs>             \
  RAJA_VERBOSE("\nacc " STR(EXEC_TYPE) " " STR(DESCRIPTOR))                \
  RAJA_INLINE acc::When<Policy, acc::GANGS, acc::WORKERS, acc::VECTORS>    \
  forallN_policy(ForallN_ACC_##EXEC_TYPE##_Tag, Body body, PArgs... pargs) \
  {                                                                        \
    using NextPolicy = typename Policy::NextPolicy;                        \
    using NextPolicyTag = typename Policy::NextPolicy::PolicyTag;          \
    PRAGMA_ACC(EXEC_TYPE, DESCRIPTOR)                                      \
    {                                                                      \
      forallN_policy<NextPolicy>(NextPolicyTag(),                          \
                                 std::forward<Body>(body),                 \
                                 std::forward<PArgs>(pargs)...);           \
    }                                                                      \
  }

FORALLN(parallel, no_ngangs, no_nworkers, no_nvectors, )
FORALLN(parallel, ngangs, no_nworkers, no_nvectors, num_gangs(Config::ngangs))
FORALLN(parallel,
        no_ngangs,
        nworkers,
        no_nvectors,
        num_workers(Config::nworkers))
FORALLN(parallel,
        ngangs,
        nworkers,
        no_nvectors,
        num_gangs(Config::ngangs) num_workers(Config::nworkers))
FORALLN(parallel,
        no_ngangs,
        no_nworkers,
        nvectors,
        vector_length(Config::nvectors))
FORALLN(parallel,
        ngangs,
        no_nworkers,
        nvectors,
        num_gangs(Config::ngangs) vector_length(Config::nvectors))
FORALLN(parallel,
        no_ngangs,
        nworkers,
        nvectors,
        num_workers(Config::nworkers) vector_length(Config::nvectors))
FORALLN(parallel,
        ngangs,
        nworkers,
        nvectors,
        num_gangs(Config::ngangs) num_workers(Config::nworkers)
            vector_length(Config::nvectors))
FORALLN(kernels, no_ngangs, no_nworkers, no_nvectors, )
FORALLN(kernels, ngangs, no_nworkers, no_nvectors, num_gangs(Config::ngangs))
FORALLN(kernels,
        no_ngangs,
        nworkers,
        no_nvectors,
        num_workers(Config::nworkers))
FORALLN(kernels,
        ngangs,
        nworkers,
        no_nvectors,
        num_gangs(Config::ngangs) num_workers(Config::nworkers))
FORALLN(kernels,
        no_ngangs,
        no_nworkers,
        nvectors,
        vector_length(Config::nvectors))
FORALLN(kernels,
        ngangs,
        no_nworkers,
        nvectors,
        num_gangs(Config::ngangs) vector_length(Config::nvectors))
FORALLN(kernels,
        no_ngangs,
        nworkers,
        nvectors,
        num_workers(Config::nworkers) vector_length(Config::nvectors))
FORALLN(kernels,
        ngangs,
        nworkers,
        nvectors,
        num_gangs(Config::ngangs) num_workers(Config::nworkers)
            vector_length(Config::nvectors))

#undef FORALLN

#undef STR_
#undef STR
#undef PRAGMA_ACC
#undef PRAGMA

}  // namespace RAJA

#endif  // closing endif for if defined(RAJA_ENABLE_OPENACC)

#endif  // closing endif for header file include guard
