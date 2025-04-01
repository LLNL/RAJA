#ifndef OPENMP_TARGET_KERNELNAME_HPP
#define OPENMP_TARGET_KERNELNAME_HPP

#include "RAJA/pattern/params/kernel_name.hpp"

namespace RAJA
{
namespace detail
{

#if defined(RAJA_ENABLE_TARGET_OPENMP)

// Init
template<typename EXEC_POL>
camp::concepts::enable_if<RAJA::type_traits::is_target_openmp_policy<EXEC_POL>>
param_init(EXEC_POL const&, Name&)
{
  // TODO: Define kernel naming
}

// Combine
template<typename EXEC_POL, typename T>
camp::concepts::enable_if<RAJA::type_traits::is_target_openmp_policy<EXEC_POL>>
param_combine(EXEC_POL const&, Name&, T& /*place holder argument*/)
{}

// Resolve
template<typename EXEC_POL>
camp::concepts::enable_if<RAJA::type_traits::is_target_openmp_policy<EXEC_POL>>
param_resolve(EXEC_POL const&, Name&)
{
  // TODO: Define kernel naming
}

#endif

}  //  namespace detail
}  //  namespace RAJA


#endif  //  NEW_REDUCE_SEQ_REDUCE_HPP
