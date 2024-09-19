#ifndef NEW_REDUCE_SYCL_REDUCE_HPP
#define NEW_REDUCE_SYCL_REDUCE_HPP

#include "RAJA/pattern/params/reducer.hpp"

namespace RAJA {
namespace expt {
namespace detail {

#if defined(RAJA_ENABLE_SYCL)

  // Init
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T, typename D, typename I, typename VType>
  camp::concepts::enable_if< type_traits::is_sycl_policy<EXEC_POL> >
  init(Reducer<OP, T, VType, D, I, true>& red) {
    red.valop_m.val = OP<T,T,T>::identity();
  }

  // Combine
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T, typename D, typename I, typename VType>
  camp::concepts::enable_if< type_traits::is_sycl_policy<EXEC_POL> >
  combine(Reducer<OP, T, VType, D, I, true>& out, const Reducer<OP, T, VType, D, I, true>& in) {
    out.valop_m.val = OP<T,T,T>{}(out.valop_m.val, in.valop_m.val);
  }

  // Resolve
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T, typename D, typename I, typename VType>
  camp::concepts::enable_if< type_traits::is_sycl_policy<EXEC_POL> >
  resolve(Reducer<OP, T, VType, D, I, true>& red) {
    *red.target = OP<T,T,T>{}(*red.target, red.valop_m.val);
    *red.passthruval = red.valop_m.val.val;
    *red.passthruindex = red.valop_m.val.loc;
  }

  // Init
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T, typename D, typename I, typename VType>
  camp::concepts::enable_if< type_traits::is_sycl_policy<EXEC_POL> >
  init(Reducer<OP, T, VType, D, I, false>& red) {
    red.valop_m.val = OP<T,T,T>::identity();
  }

  // Combine
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T, typename D, typename I, typename VType>
  camp::concepts::enable_if< type_traits::is_sycl_policy<EXEC_POL> >
  combine(Reducer<OP, T, VType, D, I, false>& out, const Reducer<OP, T, VType, D, I, false>& in) {
    out.valop_m.val = OP<T,T,T>{}(out.valop_m.val, in.valop_m.val);
  }

  // Resolve
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T, typename D, typename I, typename VType>
  camp::concepts::enable_if< type_traits::is_sycl_policy<EXEC_POL> >
  resolve(Reducer<OP, T, VType, D, I, false>& red) {
    *red.target = OP<T,T,T>{}(*red.target, red.valop_m.val);
  }

#endif

} //  namespace detail
} //  namespace expt
} //  namespace RAJA

#endif //  NEW_REDUCE_SYCL_REDUCE_HPP
