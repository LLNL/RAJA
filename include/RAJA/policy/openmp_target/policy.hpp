//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_openmp_target_HPP
#define RAJA_policy_openmp_target_HPP

#include "RAJA/policy/PolicyBase.hpp"

namespace RAJA {

namespace policy {
namespace omp {

// Max number of CUDA reduction threads per block possible.
// Required for allocating omp target data before execution policy.
// Used in target_parallel_for, aliased in target_reduce.
static constexpr int MAXNUMTHREADS = 1024;

template <unsigned int TeamSize>
struct Teams : std::integral_constant<unsigned int, TeamSize> {
};

struct Target {
};

struct Distribute {
};

struct Collapse {
};

template <size_t ThreadsPerTeam>
struct omp_target_parallel_for_exec
    : make_policy_pattern_platform_t<Policy::target_openmp,
                            Pattern::forall,
                            Platform::omp_target,
                            omp::Target,
                            omp::Teams<ThreadsPerTeam>,
                            omp::Distribute> {
};

struct omp_target_parallel_for_exec_nt
    : make_policy_pattern_platform_t<Policy::target_openmp,
                            Pattern::forall,
                            Platform::omp_target,
                            omp::Target,
                            omp::Distribute> {
};

struct omp_target_parallel_collapse_exec
    : make_policy_pattern_platform_t<Policy::target_openmp,
                            Pattern::forall,
                            Platform::omp_target,
                            omp::Target,
                            omp::Collapse> {
};

struct omp_target_reduce
    : make_policy_pattern_platform_t<Policy::target_openmp, Pattern::reduce, Platform::omp_target> {
};

///
/// WorkGroup execution policies
///
struct omp_target_work
    : make_policy_pattern_launch_platform_t<Policy::target_openmp,
                                            Pattern::workgroup_exec,
                                            Launch::sync,
                                            Platform::omp_target> {
};


}  // closing brace for omp namespace
}  // closing brace for policy namespace

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using policy::omp::omp_target_parallel_for_exec;
using policy::omp::omp_target_parallel_for_exec_nt;
using policy::omp::omp_target_reduce;
using policy::omp::omp_target_parallel_collapse_exec;
using policy::omp::omp_target_work;
#endif

} // closing brace for RAJA namespace

#endif // RAJA_policy_openmp_target_HPP
