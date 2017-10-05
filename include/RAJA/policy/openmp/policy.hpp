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

#include <type_traits>

namespace RAJA
{

namespace omp
{

struct Parallel {
};

struct Collapse {
};

struct For {
};

struct NoWait {
};

template <unsigned int ChunkSize>
struct Static : std::integral_constant<unsigned int, ChunkSize> {
};

#if defined(RAJA_ENABLE_TARGET_OPENMP)

template <unsigned int TeamSize>
struct Teams : std::integral_constant<unsigned int, TeamSize> {
};

struct Target {
};

struct Distribute {
};

#endif
}

//
//////////////////////////////////////////////////////////////////////
//
// Execution policies
//
//////////////////////////////////////////////////////////////////////
//

struct omp_for_exec
    : make_policy_pattern_t<Policy::openmp, Pattern::forall, omp::For> {
};

struct omp_for_nowait_exec
    : make_policy_pattern_launch_platform_t<Policy::openmp,
                                            Pattern::forall,
                                            Launch::undefined,
                                            Platform::host,
                                            omp::For,
                                            omp::NoWait> {
};

struct omp_for_dependence_graph
    : make_policy_pattern_launch_platform_t<Policy::openmp,
                                            Pattern::forall,
                                            Launch::undefined,
                                            Platform::host,
                                            omp::For,
                                            omp::Static<1>> {
};

template <unsigned int N>
struct omp_for_static : make_policy_pattern_launch_platform_t<Policy::openmp,
                                                              Pattern::forall,
                                                              Launch::undefined,
                                                              Platform::host,
                                                              omp::For,
                                                              omp::Static<N>> {
};


template <typename InnerPolicy>
struct omp_parallel_exec
    : make_policy_pattern_launch_platform_t<Policy::openmp,
                                            Pattern::forall,
                                            Launch::undefined,
                                            Platform::host,
                                            omp::Parallel,
                                            wrapper<InnerPolicy>> {
};

struct omp_parallel_for_exec : omp_parallel_exec<omp_for_exec> {
};

template <unsigned int N>
struct omp_parallel_for_static : omp_parallel_exec<omp_for_static<N>> {
};

///
/// Policies for applying OpenMP clauses in forallN loop nests.
///
struct omp_collapse_nowait_exec
    : make_policy_pattern_launch_platform_t<Policy::openmp,
                                            Pattern::forall,
                                            Launch::undefined,
                                            Platform::host,
                                            omp::Collapse> {
};

#if defined(RAJA_ENABLE_TARGET_OPENMP)
template <size_t Teams>
struct omp_target_parallel_for_exec
    : make_policy_pattern_t<Policy::target_openmp,
                            Pattern::forall,
                            omp::Target,
                            omp::Teams<Teams>,
                            omp::Distribute> {
};
#endif

///
/// Index set segment iteration policies
///

using omp_parallel_for_segit = omp_parallel_for_exec;

using omp_parallel_segit = omp_parallel_for_segit;

struct omp_taskgraph_segit
    : make_policy_pattern_t<Policy::openmp, Pattern::taskgraph, omp::Parallel> {
};

struct omp_taskgraph_interval_segit
    : make_policy_pattern_t<Policy::openmp, Pattern::taskgraph, omp::Parallel> {
};

///
///////////////////////////////////////////////////////////////////////
///
/// Reduction execution policies
///
///////////////////////////////////////////////////////////////////////
///

struct omp_reduce : make_policy_pattern_t<Policy::openmp, Pattern::reduce> {
};

#if defined(RAJA_ENABLE_TARGET_OPENMP)
template <size_t Teams>
struct omp_target_reduce
    : make_policy_pattern_t<Policy::target_openmp, Pattern::reduce> {
};
#endif

struct omp_reduce_ordered
    : make_policy_pattern_t<Policy::openmp, Pattern::reduce, reduce::ordered> {
};

}  // closing brace for RAJA namespace


#endif
