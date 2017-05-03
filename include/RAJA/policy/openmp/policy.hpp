#ifndef policy_openmp_HXX_
#define policy_openmp_HXX_

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
