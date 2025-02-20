#ifndef SIMD_KERNELNAME_HPP
#define SIMD_KERNELNAME_HPP

#include "RAJA/pattern/params/kernel_name.hpp"

namespace RAJA
{
namespace expt
{
namespace detail
{

// Init
template<typename EXEC_POL>
camp::concepts::enable_if<std::is_same<EXEC_POL, RAJA::simd_exec>> param_init(
    EXEC_POL const&,
    KernelName&)
{
  // TODO: Define kernel naming
}

// Combine
template<typename EXEC_POL, typename T>
RAJA_HOST_DEVICE camp::concepts::enable_if<
    std::is_same<EXEC_POL, RAJA::simd_exec>>
param_combine(EXEC_POL const&, KernelName&, T)
{}

// Resolve
template<typename EXEC_POL>
camp::concepts::enable_if<std::is_same<EXEC_POL, RAJA::simd_exec>> param_resolve(
    EXEC_POL const&,
    KernelName&)
{
  // TODO: Define kernel naming
}

}  //  namespace detail
}  //  namespace expt
}  //  namespace RAJA


#endif  //  NEW_REDUCE_SIMD_REDUCE_HPP
