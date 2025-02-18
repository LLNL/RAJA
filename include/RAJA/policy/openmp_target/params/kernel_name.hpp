#ifndef OPENMP_TARGET_KERNELNAME_HPP
#define OPENMP_TARGET_KERNELNAME_HPP

#include "RAJA/pattern/params/kernel_name.hpp"

namespace RAJA
{
namespace expt
{
namespace detail
{

#if defined(RAJA_ENABLE_TARGET_OPENMP)

// Init
template<typename EXEC_POL>
camp::concepts::enable_if<type_traits::is_target_openmp_policy<EXEC_POL>> param_init(
    KernelName&)
{
  // TODO: Define kernel naming
}

// Combine
template<typename EXEC_POL, typename T>
camp::concepts::enable_if<type_traits::is_target_openmp_policy<EXEC_POL>>
param_combine(KernelName&, T& /*place holder argument*/)
{}

// Resolve
template<typename EXEC_POL>
camp::concepts::enable_if<type_traits::is_target_openmp_policy<EXEC_POL>>
param_resolve(KernelName&)
{
  // TODO: Define kernel naming
}

#endif

}  //  namespace detail
}  //  namespace expt
}  //  namespace RAJA


#endif  //  NEW_REDUCE_SEQ_REDUCE_HPP
