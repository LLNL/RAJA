#ifndef POLICY_HPP
#define POLICY_HPP

#include <RAJA/RAJA.hpp>

#if defined(RAJA_ENABLE_CUDA)
template<typename>
struct is_cuda_policy : std::false_type {};

template<size_t BS, bool A>
struct is_cuda_policy<RAJA::cuda_exec<BS, A>> : std::true_type {};
#endif

#if defined(RAJA_ENABLE_OPENMP)
template<typename>
struct is_openmp_policy : std::false_type {};

template<>
struct is_openmp_policy<RAJA::omp_parallel_for_exec> : std::true_type {};
#endif

#endif
