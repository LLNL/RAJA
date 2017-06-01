/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA OpenMP policy definitions.
 *
 ******************************************************************************
 */

#ifndef policy_openmp_HPP
#define policy_openmp_HPP

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

#include "RAJA/policy/PolicyBase.hpp"

namespace RAJA
{

//
//////////////////////////////////////////////////////////////////////
//
// Execution policies
//
//////////////////////////////////////////////////////////////////////
//

///
/// Segment execution policies
///
template <typename InnerPolicy>
struct omp_parallel_exec : public RAJA::wrap<InnerPolicy> {
};

struct omp_for_exec : public RAJA::make_policy_pattern<RAJA::Policy::openmp,
                                                       RAJA::Pattern::forall> {
};

struct omp_parallel_for_exec : public omp_parallel_exec<omp_for_exec> {
};

template <size_t ChunkSize>
struct omp_for_static
    : public RAJA::make_policy_pattern<RAJA::Policy::openmp,
                                       RAJA::Pattern::forall> {
};

template <size_t ChunkSize>
struct omp_parallel_for_static
    : public omp_parallel_exec<omp_for_static<ChunkSize>> {
};

struct omp_for_nowait_exec
    : public RAJA::make_policy_pattern<RAJA::Policy::openmp,
                                       RAJA::Pattern::forall> {
};


///
/// Index set segment iteration policies
///
struct omp_parallel_for_segit : public omp_parallel_for_exec {
};
struct omp_parallel_segit : public omp_parallel_for_segit {
};
struct omp_taskgraph_segit
    : public RAJA::make_policy_pattern<RAJA::Policy::openmp,
                                       RAJA::Pattern::taskgraph> {
};
struct omp_taskgraph_interval_segit
    : public RAJA::make_policy_pattern<RAJA::Policy::openmp,
                                       RAJA::Pattern::taskgraph> {
};


///
/// Policies for applying OpenMP clauses in forallN loop nests.
///
struct omp_collapse_nowait_exec {
};

///
///////////////////////////////////////////////////////////////////////
///
/// Reduction execution policies
///
///////////////////////////////////////////////////////////////////////
///
struct omp_reduce : public RAJA::make_policy_pattern<RAJA::Policy::openmp,
                                                     RAJA::Pattern::reduce> {
};

struct omp_reduce_ordered : public omp_reduce {
};

}  // closing brace for RAJA namespace

#endif
