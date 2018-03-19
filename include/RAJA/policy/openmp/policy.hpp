/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA OpenMP policy definitions.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef policy_openmp_HPP
#define policy_openmp_HPP

#include "RAJA/policy/PolicyBase.hpp"

#include <type_traits>

namespace RAJA
{
namespace policy
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

struct omp_target_parallel_for_exec_nt
    : make_policy_pattern_t<Policy::target_openmp,
                            Pattern::forall,
                            omp::Target,
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

struct omp_synchronize : make_policy_pattern_launch_t<Policy::openmp,
                                                      Pattern::synchronize,
                                                      Launch::sync> {
};

}  // closing brace for omp namespace
}  // closing brace for policy namespace

using policy::omp::omp_for_exec;
using policy::omp::omp_for_nowait_exec;
using policy::omp::omp_for_static;
using policy::omp::omp_parallel_exec;
using policy::omp::omp_parallel_for_exec;
using policy::omp::omp_parallel_segit;
using policy::omp::omp_parallel_for_segit;
using policy::omp::omp_collapse_nowait_exec;
using policy::omp::omp_reduce;
using policy::omp::omp_reduce_ordered;
using policy::omp::omp_synchronize;

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using policy::omp::omp_target_parallel_for_exec;
using policy::omp::omp_target_parallel_for_exec_nt;
using policy::omp::omp_target_reduce;
#endif


///
///////////////////////////////////////////////////////////////////////
///
/// Shared memory policies
///
///////////////////////////////////////////////////////////////////////
///

using omp_shmem = seq_shmem;

}  // closing brace for RAJA namespace


#endif
