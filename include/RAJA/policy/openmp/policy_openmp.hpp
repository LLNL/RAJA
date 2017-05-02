#ifndef policy_openmp_HXX_
#define policy_openmp_HXX_

#include "RAJA/policy/PolicyFamily.hpp"

namespace RAJA
{

//
//////////////////////////////////////////////////////////////////////
//
// Execution policies
//
//////////////////////////////////////////////////////////////////////
//

struct omp_exec_base {};

///
/// Segment execution policies
///
template <typename InnerPolicy>
struct omp_parallel_exec  : public omp_exec_base {
  const PolicyFamily family = PolicyFamily::openmp;
};
struct omp_for_exec : public omp_exec_base { 
  const PolicyFamily family = PolicyFamily::openmp;
};
struct omp_parallel_for_exec : public omp_parallel_exec<omp_for_exec> {
};
template <size_t ChunkSize>
struct omp_for_static : public omp_exec_base {
  const PolicyFamily family = PolicyFamily::openmp;
};
template <size_t ChunkSize>
struct omp_parallel_for_static
    : public omp_parallel_exec<omp_for_static<ChunkSize>> {
};
struct omp_for_nowait_exec : public omp_exec_base {
  const PolicyFamily family = PolicyFamily::openmp;
};

///
/// Index set segment iteration policies
///
struct omp_parallel_for_segit : public omp_parallel_for_exec {
};
struct omp_parallel_segit : public omp_parallel_for_segit {
};
struct omp_taskgraph_segit : public omp_exec_base {
  const PolicyFamily family = PolicyFamily::openmp;
};
struct omp_taskgraph_interval_segit : public omp_exec_base {
  const PolicyFamily family = PolicyFamily::openmp;
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
