#ifndef SEQ_KERNELNAME_HPP
#define SEQ_KERNELNAME_HPP

#include "RAJA/pattern/params/kernel_name.hpp"

namespace RAJA
{
namespace expt
{
namespace detail
{

// Init
template<typename EXEC_POL>
camp::concepts::enable_if<std::is_same<EXEC_POL, RAJA::seq_exec>> param_init(
    EXEC_POL const&,
    RAJA::detail::Name&)
{
  // TODO: Define kernel naming
}

// Combine
template<typename EXEC_POL, typename T>
RAJA_HOST_DEVICE camp::concepts::enable_if<
    std::is_same<EXEC_POL, RAJA::seq_exec>>
param_combine(EXEC_POL const&, RAJA::detail::Name&, T)
{}

// Resolve
template<typename EXEC_POL>
camp::concepts::enable_if<std::is_same<EXEC_POL, RAJA::seq_exec>> param_resolve(
    EXEC_POL const&,
    RAJA::detail::Name&)
{
  // TODO: Define kernel naming
}

}  //  namespace detail
}  //  namespace expt
}  //  namespace RAJA


#endif  //  NEW_REDUCE_SEQ_REDUCE_HPP
