#ifndef CUDA_KERNELNAME_HPP
#define CUDA_KERNELNAME_HPP

#if defined(RAJA_CUDA_ACTIVE)

#include <cuda.h>
#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#include "RAJA/pattern/params/kernel_name.hpp"

#if defined(RAJA_ENABLE_NV_TOOLS_EXT)
#include <nvToolsExt.h>
#endif

namespace RAJA
{
namespace expt
{
namespace detail
{

// Init
template<typename EXEC_POL>
camp::concepts::enable_if<RAJA::type_traits::is_cuda_policy<EXEC_POL>>
param_init(EXEC_POL const&,
           RAJA::detail::Name& kn,
           const RAJA::cuda::detail::cudaInfo&)
{
#if defined(RAJA_ENABLE_NV_TOOLS_EXT)
  if (kn.name != nullptr && expt::detail::RAJA_caliper_profile == false)
  {
    nvtxRangePush(kn.name);
  }
#else
  RAJA_UNUSED_VAR(kn);
#endif
}

// Combine
template<typename EXEC_POL>
RAJA_HOST_DEVICE camp::concepts::enable_if<
    RAJA::type_traits::is_cuda_policy<EXEC_POL>>
param_combine(EXEC_POL const&, RAJA::detail::Name&)
{}

// Resolve
template<typename EXEC_POL>
camp::concepts::enable_if<RAJA::type_traits::is_cuda_policy<EXEC_POL>>
param_resolve(EXEC_POL const&,
              RAJA::detail::Name& kn,
              const RAJA::cuda::detail::cudaInfo&)
{
#if defined(RAJA_ENABLE_NV_TOOLS_EXT)
  if (kn.name != nullptr && expt::detail::RAJA_caliper_profile == false)
  {
    nvtxRangePop();
  }
#else
  RAJA_UNUSED_VAR(kn);
#endif
}

}  //  namespace detail
}  //  namespace expt
}  //  namespace RAJA

#endif

#endif  //  NEW_REDUCE_CUDA_REDUCE_HPP
