#ifndef NEW_REDUCE_OMP_REDUCE_HPP
#define NEW_REDUCE_OMP_REDUCE_HPP

#include "RAJA/pattern/params/reducer.hpp"

namespace RAJA {
namespace expt {
namespace detail {

#if defined(RAJA_ENABLE_OPENMP)

/*
  // Init
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T, typename VType,
           std::enable_if_t<std::is_same<T, RAJA::expt::ValLoc<int,RAJA::Index_type>>::value>* = nullptr >
  camp::concepts::enable_if< type_traits::is_openmp_policy<EXEC_POL> >
  init(Reducer<OP, T, VType>& red) {
    red.val.val = OP<T,T,T>::identity();
  }

  // Combine
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T, typename VType,
           std::enable_if_t<std::is_same<T, RAJA::expt::ValLoc<int,RAJA::Index_type>>::value>* = nullptr >
  camp::concepts::enable_if< type_traits::is_openmp_policy<EXEC_POL> >
  combine(Reducer<OP, T, VType>& out, const Reducer<OP, T, VType>& in) {
    out.val.val = OP<T,T,T>{}(out.val.val, in.val.val);
  }

  // Resolve
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T, typename VType,
           std::enable_if_t<std::is_same<T, RAJA::expt::ValLoc<int,RAJA::Index_type>>::value>* = nullptr >
  camp::concepts::enable_if< type_traits::is_openmp_policy<EXEC_POL> >
  resolve(Reducer<OP, T, VType>& red) {
  //resolve(Reducer<OP, T, ValOp<ValLoc<T,U>,OP>>& red) {
    *red.target = OP<T,T,T>{}(*red.target, red.val.val);
  }

  // Init
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T, typename VType,
           std::enable_if_t<std::is_integral<T>::value || std::is_floating_point<T>::value>* = nullptr >
  //template<typename EXEC_POL, typename OP, typename T>
  camp::concepts::enable_if< type_traits::is_openmp_policy<EXEC_POL> >
  init(Reducer<OP, T, VType>& red) {
  //init(Reducer<OP, T>& red) {
    red.val.val = OP<T,T,T>::identity();
  }

  // Combine
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T, typename VType,
           std::enable_if_t<std::is_integral<T>::value || std::is_floating_point<T>::value>* = nullptr >
  //template<typename EXEC_POL, typename OP, typename T>
  camp::concepts::enable_if< type_traits::is_openmp_policy<EXEC_POL> >
  combine(Reducer<OP, T, VType>& out, const Reducer<OP, T, VType>& in) {
  //combine(Reducer<OP, T>& out, const Reducer<OP, T>& in) {
    out.val.val = OP<T,T,T>{}(out.val.val, in.val.val);
  }

  // Resolve
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T, typename VType,
           std::enable_if_t<std::is_integral<T>::value || std::is_floating_point<T>::value>* = nullptr >
  //template<typename EXEC_POL, typename OP, typename T>
  camp::concepts::enable_if< type_traits::is_openmp_policy<EXEC_POL> >
  resolve(Reducer<OP, T, VType>& red) {
  //resolve(Reducer<OP, T>& red) {
    *red.target = OP<T,T,T>{}(*red.target, red.val.val);
  }
*/

  // Init
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T, typename VType>
  camp::concepts::enable_if< type_traits::is_openmp_policy<EXEC_POL> >
  init(Reducer<OP, T, VType>& red) {
    red.val.val = OP<T,T,T>::identity();
  }

  // Combine
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T, typename VType>
  camp::concepts::enable_if< type_traits::is_openmp_policy<EXEC_POL> >
  combine(Reducer<OP, T, VType>& out, const Reducer<OP, T, VType>& in) {
    out.val.val = OP<T,T,T>{}(out.val.val, in.val.val);
  }

  // Resolve
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T, typename VType>
  camp::concepts::enable_if< type_traits::is_openmp_policy<EXEC_POL> >
  resolve(Reducer<OP, T, VType>& red) {
    *red.target = OP<T,T,T>{}(*red.target, red.val.val);
  }

#endif

} //  namespace detail
} //  namespace expt
} //  namespace RAJA

#endif //  NEW_REDUCE_OMP_REDUCE_HPP
