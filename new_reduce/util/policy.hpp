#ifndef POLICY_HPP
#define POLICY_HPP

#include <RAJA/RAJA.hpp>

#if defined(RAJA_ENABLE_CUDA)
template<typename>
struct is_cuda_policy : std::false_type {};

template<size_t BS, bool A>
struct is_cuda_policy<RAJA::cuda_exec<BS, A>> : std::true_type {};
#endif

#endif
