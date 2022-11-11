#ifndef CUDA_KERNELNAME_HPP
#define CUDA_KERNELNAME_HPP

//#include "../util/policy.hpp"

#if defined(RAJA_CUDA_ACTIVE)

#include <cuda.h>
#include "RAJA/pattern/params/kernel_name.hpp"

namespace RAJA {
namespace expt {
namespace detail {

  // Init
  template<typename EXEC_POL>
  camp::concepts::enable_if< type_traits::is_cuda_policy<EXEC_POL> >
  init(KernelName& kn, const RAJA::cuda::detail::cudaInfo & cs)
  {
#if defined(RAJA_ENABLE_NV_TOOLS_EXT)
    nvtxRangePush(kn.name);
#endif
  }

  // Combine
  template<typename EXEC_POL>
  RAJA_HOST_DEVICE
  camp::concepts::enable_if< type_traits::is_cuda_policy<EXEC_POL> >
  combine(KernelName&) {}

  // Resolve
  template<typename EXEC_POL>
  camp::concepts::enable_if< type_traits::is_cuda_policy<EXEC_POL> >
  resolve(KernelName&)
  {
#if defined(RAJA_ENABLE_NV_TOOLS_EXT)
    nvtxRangePop();
#endif
  }

} //  namespace detail
} //  namespace expt
} //  namespace RAJA

#endif

#endif //  NEW_REDUCE_CUDA_REDUCE_HPP
