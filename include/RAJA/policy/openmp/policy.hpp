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
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef policy_openmp_HPP
#define policy_openmp_HPP

#include <type_traits>

#include "RAJA/policy/PolicyBase.hpp"

#if defined(RAJA_COMPILER_MSVC)
typedef enum omp_sched_t { 
    // schedule kinds 
    omp_sched_static = 0x1, 
    omp_sched_dynamic = 0x2, 
    omp_sched_guided = 0x3, 
    omp_sched_auto = 0x4, 
    
    // schedule modifier 
    omp_sched_monotonic = 0x80000000u 
} omp_sched_t;
#else
#include <omp.h>
#endif

namespace RAJA
{
namespace policy
{
namespace omp
{

namespace internal
{
    struct ScheduleTag {};

    template <omp_sched_t Sched, int Chunk>
    struct Schedule : public ScheduleTag {
        constexpr static omp_sched_t schedule = Sched;
        constexpr static int chunk_size = Chunk;
    };
}  // namespace internal

//
//////////////////////////////////////////////////////////////////////
//
// Clauses/Keywords
//
//////////////////////////////////////////////////////////////////////
//

struct Parallel {
};

struct For {
};

struct NoWait {
};

static constexpr int default_chunk_size = -1;

struct Auto : private internal::Schedule<omp_sched_auto, default_chunk_size>{
};

template <int ChunkSize = default_chunk_size>
struct Static : public internal::Schedule<omp_sched_static, ChunkSize> {
};

template <int ChunkSize = default_chunk_size>
using Dynamic = internal::Schedule<omp_sched_dynamic, ChunkSize>;

template <int ChunkSize = default_chunk_size>
using Guided = internal::Schedule<omp_sched_guided, ChunkSize>;

struct Runtime : private internal::Schedule<static_cast<omp_sched_t>(-1), default_chunk_size> {
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

template <typename Sched>
struct omp_for_nowait_schedule_exec : make_policy_pattern_launch_platform_t<Policy::openmp,
                                                              Pattern::forall,
                                                              Launch::undefined,
                                                              Platform::host,
                                                              omp::For,
                                                              omp::NoWait,
                                                              Sched> {
    static_assert(std::is_base_of<::RAJA::policy::omp::internal::ScheduleTag, Sched>::value,
        "Schedule must be one of: Auto|Runtime|Static|Dynamic|Guided");
};


template <typename Sched>
struct omp_for_schedule_exec : make_policy_pattern_launch_platform_t<Policy::openmp,
                                                              Pattern::forall,
                                                              Launch::undefined,
                                                              Platform::host,
                                                              omp::For,
                                                              Sched> {
    static_assert(std::is_base_of<::RAJA::policy::omp::internal::ScheduleTag, Sched>::value,
        "Schedule must be one of: Auto|Runtime|Static|Dynamic|Guided");
};

using omp_for_exec = omp_for_schedule_exec<Auto>;

template <int ChunkSize = default_chunk_size>
using omp_for_static_exec = omp_for_schedule_exec<omp::Static<ChunkSize>>;

template <int ChunkSize = default_chunk_size>
using omp_for_dynamic_exec = omp_for_schedule_exec<omp::Dynamic<ChunkSize>>;

template <int ChunkSize = default_chunk_size>
using omp_for_guided_exec = omp_for_schedule_exec<omp::Guided<ChunkSize>>;


using omp_for_nowait_exec = omp_for_nowait_schedule_exec<Auto>;

template <int ChunkSize = default_chunk_size>
using omp_for_nowait_static_exec = omp_for_nowait_schedule_exec<omp::Static<ChunkSize>>;

template <int ChunkSize = default_chunk_size>
using omp_for_nowait_dynamic_exec = omp_for_nowait_schedule_exec<omp::Dynamic<ChunkSize>>;

template <int ChunkSize = default_chunk_size>
using omp_for_nowait_guided_exec = omp_for_nowait_schedule_exec<omp::Guided<ChunkSize>>;


template <typename InnerPolicy>
using omp_parallel_exec = make_policy_pattern_launch_platform_t<Policy::openmp,
                                            Pattern::forall,
                                            Launch::undefined,
                                            Platform::host,
                                            omp::Parallel,
                                            wrapper<InnerPolicy>>;

using omp_parallel_for_exec = omp_parallel_exec<omp_for_exec>;

template <int ChunkSize = default_chunk_size>
using omp_parallel_for_static_exec = omp_parallel_exec<omp_for_schedule_exec<omp::Static<ChunkSize>> >;

template <int ChunkSize = default_chunk_size>
using omp_parallel_for_dynamic_exec = omp_parallel_exec<omp_for_schedule_exec<omp::Dynamic<ChunkSize>> >;

template <int ChunkSize = default_chunk_size>
using omp_parallel_for_guided_exec = omp_parallel_exec<omp_for_schedule_exec<omp::Guided<ChunkSize>> >;


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
/// WorkGroup execution policies
///
struct omp_work : make_policy_pattern_launch_platform_t<Policy::openmp,
                                                        Pattern::workgroup_exec,
                                                        Launch::sync,
                                                        Platform::host> {
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


//
// Type aliases to simplify common omp parallel for loop execution
//
using policy::omp::omp_parallel_for_exec;
using policy::omp::omp_parallel_for_static_exec;
using policy::omp::omp_parallel_for_dynamic_exec;
using policy::omp::omp_parallel_for_guided_exec;

//
// Type aliases for omp parallel for iteration over indexset segments
//
using policy::omp::omp_parallel_for_segit;
using policy::omp::omp_parallel_segit;

//
// Type alias for omp parallel region containing an inner omp for loop 
// execution policy. Inner policy types follow.
//
using policy::omp::omp_parallel_exec;

//
// Type aliases for omp for and omp for nowait loop execution within
// an omp_parallel_exec construct
//
using policy::omp::omp_for_exec;
using policy::omp::omp_for_nowait_exec;

//
// Type aliases for omp for and omp for nowait loop execution with a 
// scheduling policy within an omp_parallel_exec construct
// Scheduling policies are near the top of this file and include:
// RAJA::policy::omp::{Auto, Static, Dynamic, Guided, Runtime}
//
// Helper aliases to make usage less verbose for common use cases follow these.
//
using policy::omp::omp_for_schedule_exec;
using policy::omp::omp_for_nowait_schedule_exec;

//
// Type aliases for omp for and omp for nowait loop execution with a 
// static scheduling policy within an omp_parallel_exec construct
//
using policy::omp::omp_for_static_exec;
using policy::omp::omp_for_nowait_static_exec;
using policy::omp::omp_for_dynamic_exec;
using policy::omp::omp_for_nowait_dynamic_exec;
using policy::omp::omp_for_guided_exec;
using policy::omp::omp_for_nowait_guided_exec;

//
// Type aliases for omp parallel region
//
using policy::omp::omp_parallel_region;

//
// Type aliases for omp reductions
//
using policy::omp::omp_reduce;
using policy::omp::omp_reduce_ordered;

//
// Type aliases for omp reductions
//
using policy::omp::omp_synchronize;

using policy::omp::omp_work;

}  // namespace RAJA

#endif
