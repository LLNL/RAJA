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
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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

#include <type_traits>

#include "RAJA/policy/PolicyBase.hpp"

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


//
//////////////////////////////////////////////////////////////////////
//
// Execution policies
//
//////////////////////////////////////////////////////////////////////
//

struct omp_parallel_region
    : make_policy_pattern_launch_platform_t<Policy::openmp,
                                            Pattern::region,
                                            Launch::undefined,
                                            Platform::host> {
};

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


struct omp_reduce_ordered
    : make_policy_pattern_t<Policy::openmp, Pattern::reduce, reduce::ordered> {
};

struct omp_synchronize : make_policy_pattern_launch_t<Policy::openmp,
                                                      Pattern::synchronize,
                                                      Launch::sync> {
};

}  // namespace omp
}  // namespace policy

using policy::omp::omp_for_exec;
using policy::omp::omp_for_nowait_exec;
using policy::omp::omp_for_static;
using policy::omp::omp_parallel_exec;
using policy::omp::omp_parallel_for_exec;
using policy::omp::omp_parallel_for_segit;
using policy::omp::omp_parallel_region;
using policy::omp::omp_parallel_segit;
using policy::omp::omp_reduce;
using policy::omp::omp_reduce_ordered;
using policy::omp::omp_synchronize;




}  // namespace RAJA


#endif
