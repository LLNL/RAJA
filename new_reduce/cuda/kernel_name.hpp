#ifndef NEW_REDUCE_CUDA_KERNEL_NAME_HPP
#define NEW_REDUCE_CUDA_KERNEL_NAME_HPP


#if defined(RAJA_ENABLE_CUDA)

#include <cuda.h>
#include "../util/policy.hpp"

namespace detail
{

  template<typename EXEC_POL>
  camp::concepts::enable_if< is_cuda_policy< EXEC_POL > >
  init(KernelName kn, const RAJA::cuda::detail::cudaInfo &) {}

  template<typename EXEC_POL>
  RAJA_HOST_DEVICE camp::concepts::enable_if< is_cuda_policy< EXEC_POL > >
  combine(KernelName kn) {}

  template<typename EXEC_POL>
  camp::concepts::enable_if< is_cuda_policy< EXEC_POL > >
  resolve(KernelName kn) {}
} // namespace detail

#endif

#endif //  NEW_REDUCE_CUDA_KERNEL_NAME_HPP
