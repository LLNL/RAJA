#ifndef NEW_REDUCE_HIP_KERNEL_NAME_HPP
#define NEW_REDUCE_HIP_KERNEL_NAME_HPP


#if defined(RAJA_ENABLE_HIP)

//#include <hip.h>
#include "../util/policy.hpp"

namespace detail
{

  template<typename EXEC_POL>
  camp::concepts::enable_if< is_hip_policy< EXEC_POL > >
  init(KernelName kn, const RAJA::hip::detail::hipInfo &) {
#if defined(RAJA_ENABLE_NV_TOOLS_EXT)
    nvtxRangePushA(kn.name);
#endif
  }

  template<typename EXEC_POL>
  RAJA_HOST_DEVICE camp::concepts::enable_if< is_hip_policy< EXEC_POL > >
  combine(KernelName kn) {}

  template<typename EXEC_POL>
  camp::concepts::enable_if< is_hip_policy< EXEC_POL > >
  resolve(KernelName kn) {
#if defined(RAJA_ENABLE_NV_TOOLS_EXT)
    nvtxRangePop();
#endif
  }
} // namespace detail

#endif

#endif //  NEW_REDUCE_HIP_KERNEL_NAME_HPP
