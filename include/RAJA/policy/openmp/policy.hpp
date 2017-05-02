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
struct omp_parallel_exec : public wrap_policy<InnerPolicy> {
};
struct omp_for_exec : public forall_policy {
};
struct omp_parallel_for_exec : public omp_parallel_exec<omp_for_exec> {
};
template <size_t ChunkSize>
struct omp_for_static : public forall_policy {
};
template <size_t ChunkSize>
struct omp_parallel_for_static
    : public omp_parallel_exec<omp_for_static<ChunkSize>> {
};
struct omp_for_nowait_exec : public forall_policy {
};

///
/// Index set segment iteration policies
///
struct omp_parallel_for_segit : public omp_parallel_for_exec {
};
struct omp_parallel_segit : public omp_parallel_for_segit {
};
struct omp_taskgraph_segit : public taskgraph_policy {
};
struct omp_taskgraph_interval_segit : public taskgraph_policy {
};

///
///////////////////////////////////////////////////////////////////////
///
/// Reduction execution policies
///
///////////////////////////////////////////////////////////////////////
///
struct omp_reduce : public reduce_policy {
};

struct omp_reduce_ordered : public reduce_policy {
};

}  // closing brace for RAJA namespace

#endif
