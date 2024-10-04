#ifndef NEW_REDUCE_OMP_REDUCE_HPP
#define NEW_REDUCE_OMP_REDUCE_HPP

#include "RAJA/pattern/params/reducer.hpp"

namespace RAJA {
namespace expt {
namespace detail {

#if defined(RAJA_ENABLE_OPENMP)

  // Init
  template<typename EXEC_POL, typename OP, typename T, typename VType>
  camp::concepts::enable_if< type_traits::is_openmp_policy<EXEC_POL> >
  init(Reducer<OP, T, VType>& red) {
    red.m_valop.val = OP::identity();
  }

  // Combine
  template<typename EXEC_POL, typename OP, typename T, typename VType>
  camp::concepts::enable_if< type_traits::is_openmp_policy<EXEC_POL> >
  combine(Reducer<OP, T, VType>& out, const Reducer<OP, T, VType>& in) {
    out.m_valop.val = OP{}(out.m_valop.val, in.m_valop.val);
  }

  // Resolve
  template<typename EXEC_POL, typename OP, typename T, typename VType>
  camp::concepts::enable_if< type_traits::is_openmp_policy<EXEC_POL> >
  resolve(Reducer<OP, T, VType>& red) {
    red.setTarget(OP{}(*red.target, red.m_valop.val));
  }

#endif

} //  namespace detail
} //  namespace expt
} //  namespace RAJA

#endif //  NEW_REDUCE_OMP_REDUCE_HPP
