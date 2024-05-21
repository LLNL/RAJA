#ifndef NEW_REDUCE_SEQ_REDUCE_HPP
#define NEW_REDUCE_SEQ_REDUCE_HPP

#include "RAJA/pattern/params/reducer.hpp"

namespace RAJA {
namespace expt {
namespace detail {

  // Init
  //template<typename EXEC_POL, typename OP, typename Type, typename minmax, template <typename, typename> class T = RefLoc<Type, minmax>>
  template<typename EXEC_POL, typename OP, template <typename Type, typename minmax> class T, typename Type, typename minmax>
  camp::concepts::enable_if< std::is_same< EXEC_POL, RAJA::seq_exec> >
  init(Reducer<OP, Type>& red) {
  //init(ReducerRef<OP, T<Type, minmax>>& red) {
    red.val = OP::identity();
  }
  // Combine
  template<typename EXEC_POL, typename OP, template <typename Type, typename minmax> class T, typename Type, typename minmax>
  //template<typename EXEC_POL, typename OP, template <typename, typename> typename T, typename Type, typename minmax>
  camp::concepts::enable_if< std::is_same< EXEC_POL, RAJA::seq_exec> >
  combine(Reducer<OP, Type>& out, const Reducer<OP, Type>& in) {
    out.val = OP{}(out.val, in.val);
  }
  // Resolve
  template<typename EXEC_POL, typename OP, template <typename Type, typename minmax> class T, typename Type, typename minmax>
  //template<typename EXEC_POL, typename OP, template <typename, typename> typename T, typename Type, typename minmax>
  camp::concepts::enable_if< std::is_same< EXEC_POL, RAJA::seq_exec> >
  resolve(Reducer<OP, Type>& red) {
    *red.target = OP{}(*red.target, red.val);
  }

  // Init
  template<typename EXEC_POL, typename OP, typename T>
  camp::concepts::enable_if< std::is_same< EXEC_POL, RAJA::seq_exec> >
  init(Reducer<OP, T>& red) {
    red.val = OP::identity();
  }
  // Combine
  template<typename EXEC_POL, typename OP, typename T>
  camp::concepts::enable_if< std::is_same< EXEC_POL, RAJA::seq_exec> >
  combine(Reducer<OP, T>& out, const Reducer<OP, T>& in) {
    out.val = OP{}(out.val, in.val);
  }
  // Resolve
  template<typename EXEC_POL, typename OP, typename T>
  camp::concepts::enable_if< std::is_same< EXEC_POL, RAJA::seq_exec> >
  resolve(Reducer<OP, T>& red) {
    *red.target = OP{}(*red.target, red.val);
  }

} //  namespace detail
} //  namespace expt
} //  namespace RAJA

#endif //  NEW_REDUCE_SEQ_REDUCE_HPP
