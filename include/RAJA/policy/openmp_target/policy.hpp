#ifndef RAJA_policy_openmp_target_HPP
#define RAJA_policy_openmp_target_HPP

#include "RAJA/policy/PolicyBase.hpp"

namespace RAJA {

namespace policy {
namespace omp {

template <unsigned int TeamSize>
struct Teams : std::integral_constant<unsigned int, TeamSize> {
};

struct Target {
};

struct Distribute {
};

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

struct omp_target_parallel_collapse_exec
    : make_policy_pattern_t<Policy::target_openmp,
                            Pattern::forall,
                            omp::Target,
                            omp::Collapse> {
};

template <size_t Teams>
struct omp_target_reduce
    : make_policy_pattern_t<Policy::target_openmp, Pattern::reduce> {
};


}  // closing brace for omp namespace
}  // closing brace for policy namespace

#if defined(RAJA_ENABLE_TARGET_OPENMP)
using policy::omp::omp_target_parallel_for_exec;
using policy::omp::omp_target_parallel_for_exec_nt;
using policy::omp::omp_target_reduce;
using policy::omp::omp_target_parallel_collapse_exec;
#endif

} // closing brace for RAJA namespace

#endif // RAJA_policy_openmp_target_HPP
