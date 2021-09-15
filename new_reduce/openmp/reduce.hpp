#ifndef PROTO_NEW_REDUCE_OMP_REDUCE_HPP
#define PROTO_NEW_REDUCE_OMP_REDUCE_HPP

//#include "../util/policy.hpp"

namespace detail {

#if defined(RAJA_ENABLE_OPENMP)

  // Init
  template<typename EXEC_POL, typename OP, typename T>
  camp::concepts::enable_if< RAJA::type_traits::is_openmp_policy<EXEC_POL> >
  init(Reducer<OP, T>& red) {
    red.val = Reducer<OP,T>::op::identity();
  }

  // Combine
  template<typename EXEC_POL, typename OP, typename T>
  camp::concepts::enable_if< RAJA::type_traits::is_openmp_policy<EXEC_POL> >
  combine(Reducer<OP, T>& out, const Reducer<OP, T>& in) {
    out.val = typename Reducer<OP,T>::op{}(out.val, in.val);
  }

  // Resolve
  template<typename EXEC_POL, typename OP, typename T>
  camp::concepts::enable_if< RAJA::type_traits::is_openmp_policy<EXEC_POL> >
  resolve(Reducer<OP, T>& red) {
    *red.target = red.val;
  }

#endif

} //  namespace detail

#endif //  PROTO_NEW_REDUCE_OMP_REDUCE_HPP
