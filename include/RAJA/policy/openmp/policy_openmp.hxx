#ifndef policy_openmp_HXX_
#define policy_openmp_HXX_

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
struct omp_parallel_exec {
};
struct omp_for_exec {
};
struct omp_parallel_for_exec : public omp_parallel_exec<omp_for_exec> {
};
template <size_t ChunkSize>
struct omp_for_static {
};
template <size_t ChunkSize>
struct omp_parallel_for_static
    : public omp_parallel_exec<omp_for_static<ChunkSize>> {
};
struct omp_for_nowait_exec {
};

///
/// Index set segment iteration policies
///
struct omp_parallel_for_segit : public omp_parallel_for_exec {
};
struct omp_parallel_segit : public omp_parallel_for_segit {
};
struct omp_taskgraph_segit {
};
struct omp_taskgraph_interval_segit {
};

///
///////////////////////////////////////////////////////////////////////
///
/// Reduction execution policies
///
///////////////////////////////////////////////////////////////////////
///
struct omp_reduce {
};

struct omp_reduce_ordered {
};

}  // closing brace for RAJA namespace

#endif