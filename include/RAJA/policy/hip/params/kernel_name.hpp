#ifndef HIP_KERNELNAME_HPP
#define HIP_KERNELNAME_HPP

#if defined(RAJA_HIP_ACTIVE)

#include "RAJA/policy/hip/MemUtils_HIP.hpp"
#include "RAJA/pattern/params/kernel_name.hpp"

#if defined(RAJA_ENABLE_ROCTX)
#include "hip/hip_runtime_api.h"
#include "roctx.h"
#endif

namespace RAJA
{
namespace expt
{
namespace detail
{

// Init
template<typename EXEC_POL>
camp::concepts::enable_if<RAJA::type_traits::is_hip_policy<EXEC_POL>>
param_init(EXEC_POL const&,
           RAJA::detail::Name& kn,
           const RAJA::hip::detail::hipInfo&)
{
#if defined(RAJA_ENABLE_ROCTX) && !defined(RAJA_ENABLE_CALIPER)
  if (kn.name != nullptr)
  {
    roctxRangePush(kn.name);
  }
#else
  RAJA_UNUSED_VAR(kn);
#endif
}

// Combine
template<typename EXEC_POL>
RAJA_HOST_DEVICE camp::concepts::enable_if<
    RAJA::type_traits::is_hip_policy<EXEC_POL>>
param_combine(EXEC_POL const&, RAJA::detail::Name&)
{}

// Resolve
template<typename EXEC_POL>
camp::concepts::enable_if<RAJA::type_traits::is_hip_policy<EXEC_POL>>
param_resolve(EXEC_POL const&,
              RAJA::detail::Name&,
              const RAJA::hip::detail::hipInfo&)
{
#if defined(RAJA_ENABLE_ROCTX) && !defined(RAJA_ENABLE_CALIPER)
  if (kn.name != nullptr)
  {
    roctxRangePop();
  }
#endif
}

}  //  namespace detail
}  //  namespace expt
}  //  namespace RAJA

#endif

#endif  //  NEW_REDUCE_HIP_REDUCE_HPP
