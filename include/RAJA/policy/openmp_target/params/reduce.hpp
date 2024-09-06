#ifndef NEW_REDUCE_OMP_TARGET_REDUCE_HPP
#define NEW_REDUCE_OMP_TARGET_REDUCE_HPP

#include "RAJA/pattern/params/reducer.hpp"

namespace RAJA {
namespace expt {
namespace detail {

#if defined(RAJA_ENABLE_TARGET_OPENMP)

  // Init
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T>
  camp::concepts::enable_if< type_traits::is_target_openmp_policy<EXEC_POL> >
  init(Reducer<OP, T, ValOp<T,OP>>& red) {
    red.val.val = OP<T,T,T>::identity();
  }

  // Combine
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T>
  camp::concepts::enable_if< type_traits::is_target_openmp_policy<EXEC_POL> >
  combine(Reducer<OP, T, ValOp<T,OP>>& out, const Reducer<OP, T, ValOp<T,OP>>& in) {
    out.val.val = OP<T,T,T>{}(out.val.val, in.val.val);
  }

  // Resolve
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T>
  camp::concepts::enable_if< type_traits::is_target_openmp_policy<EXEC_POL> >
  resolve(Reducer<OP, T, ValOp<T,OP>>& red) {
    red.target->val = OP<T,T,T>{}(red.target->val, red.val.val);
  }

//  // Init
//  template<typename EXEC_POL, typename OP, typename T>
//  camp::concepts::enable_if< type_traits::is_target_openmp_policy<EXEC_POL> >
//  init(Reducer<OP, T>& red) {
//    red.val = OP::identity();
//  }
//
//  // Combine
//  template<typename EXEC_POL, typename OP, typename T>
//  camp::concepts::enable_if< type_traits::is_target_openmp_policy<EXEC_POL> >
//  combine(Reducer<OP, T>& out, const Reducer<OP, T>& in) {
//    out.val = OP{}(out.val, in.val);
//  }
//
//  // Resolve
//  template<typename EXEC_POL, typename OP, typename T>
//  camp::concepts::enable_if< type_traits::is_target_openmp_policy<EXEC_POL> >
//  resolve(Reducer<OP, T>& red) {
//    *red.target = OP{}(*red.target, red.val);
//  }

#endif

} //  namespace detail
} //  namespace expt
} //  namespace RAJA

#endif //  NEW_REDUCE_OMP_REDUCE_HPP
