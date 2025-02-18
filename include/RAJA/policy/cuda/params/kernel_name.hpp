#ifndef CUDA_KERNELNAME_HPP
#define CUDA_KERNELNAME_HPP

#if defined(RAJA_CUDA_ACTIVE)

#include <cuda.h>
#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#include "RAJA/pattern/params/kernel_name.hpp"

namespace RAJA
{
namespace expt
{
namespace detail
{

// Init
template<typename EXEC_POL>
camp::concepts::enable_if<type_traits::is_cuda_policy<EXEC_POL>> param_init(
    KernelName& kn,
    const RAJA::cuda::detail::cudaInfo&)
{
#if defined(RAJA_ENABLE_NV_TOOLS_EXT)
  nvtxRangePush(kn.name);
#else
  RAJA_UNUSED_VAR(kn);
#endif
}

// Combine
template<typename EXEC_POL>
RAJA_HOST_DEVICE camp::concepts::enable_if<
    type_traits::is_cuda_policy<EXEC_POL>>
param_combine(KernelName&)
{}

// Resolve
template<typename EXEC_POL>
camp::concepts::enable_if<type_traits::is_cuda_policy<EXEC_POL>> param_resolve(
    KernelName&,
    const RAJA::cuda::detail::cudaInfo&)
{
#if defined(RAJA_ENABLE_NV_TOOLS_EXT)
  nvtxRangePop();
#endif
}

}  //  namespace detail
}  //  namespace expt
}  //  namespace RAJA

#endif

#endif  //  NEW_REDUCE_CUDA_REDUCE_HPP
