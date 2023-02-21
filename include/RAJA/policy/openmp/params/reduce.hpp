#ifndef NEW_REDUCE_OMP_REDUCE_HPP
#define NEW_REDUCE_OMP_REDUCE_HPP

#include "RAJA/pattern/params/reducer.hpp"

namespace RAJA {
namespace expt {
namespace detail {

#if defined(RAJA_ENABLE_OPENMP)

  // Init
  template<typename EXEC_POL, typename OP, typename T>
  camp::concepts::enable_if< type_traits::is_openmp_policy<EXEC_POL> >
  init(Reducer<OP, T>& red) {
    red.val = OP::identity();
  }

  // Combine
  template<typename EXEC_POL, typename OP, typename T>
  camp::concepts::enable_if< type_traits::is_openmp_policy<EXEC_POL> >
  combine(Reducer<OP, T>& out, const Reducer<OP, T>& in) {
    out.val = OP{}(out.val, in.val);
  }

  // Resolve
  template<typename EXEC_POL, typename OP, typename T>
  camp::concepts::enable_if< type_traits::is_openmp_policy<EXEC_POL> >
  resolve(Reducer<OP, T>& red) {
    *red.target = OP{}(red.val, *red.target);
  }

#endif

} //  namespace detail
} //  namespace expt
} //  namespace RAJA

#endif //  NEW_REDUCE_OMP_REDUCE_HPP
