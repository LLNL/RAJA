#ifndef OPENMP_KERNELNAME_HPP
#define OPENMP_KERNELNAME_HPP

#include "RAJA/pattern/params/kernel_name.hpp"

namespace RAJA {
namespace expt {
namespace detail {

  // Init
  template<typename EXEC_POL>
  camp::concepts::enable_if< type_traits::is_openmp_policy<EXEC_POL> >
  init(KernelName& kn)
  {
    //TODO: Define kernel naming
  }

  // Combine
  template<typename EXEC_POL>
  RAJA_HOST_DEVICE
  camp::concepts::enable_if< type_traits::is_openmp_policy<EXEC_POL> >
  combine(KernelName&) {}

  // Resolve
  template<typename EXEC_POL>
  camp::concepts::enable_if< type_traits::is_openmp_policy<EXEC_POL> >
  resolve(KernelName&)
  {
    //TODO: Define kernel naming
  }

} //  namespace detail
} //  namespace expt
} //  namespace RAJA


#endif //  NEW_REDUCE_SEQ_REDUCE_HPP
