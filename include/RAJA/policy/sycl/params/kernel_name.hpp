#ifndef SYCL_KERNELNAME_HPP
#define SYCL_KERNELNAME_HPP

#include "RAJA/pattern/params/kernel_name.hpp"

namespace RAJA {
namespace expt {
namespace detail {

#if defined(RAJA_ENABLE_SYCL)  
  
  // Init
  template<typename EXEC_POL>
  camp::concepts::enable_if< type_traits::is_sycl_policy<EXEC_POL> >
  init(KernelName&)
  {
    //TODO: Define kernel naming
  }

  // Combine
  template<typename EXEC_POL, typename T>
  camp::concepts::enable_if< type_traits::is_sycl_policy<EXEC_POL> >
  SYCL_EXTERNAL
  combine(KernelName&, T) {}

  // Resolve
  template<typename EXEC_POL>
  camp::concepts::enable_if< type_traits::is_sycl_policy<EXEC_POL> >
  resolve(KernelName&)
  {
    //TODO: Define kernel naming
  }

#endif  

} //  namespace detail
} //  namespace expt
} //  namespace RAJA


#endif //  NEW_REDUCE_SYCL_REDUCE_HPP
