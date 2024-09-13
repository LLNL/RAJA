#ifndef NEW_REDUCE_SEQ_REDUCE_HPP
#define NEW_REDUCE_SEQ_REDUCE_HPP

#include "RAJA/pattern/params/reducer.hpp"

namespace RAJA {
namespace expt {
namespace detail {

  // Init
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T, typename VType>
  camp::concepts::enable_if< std::is_same< EXEC_POL, RAJA::seq_exec> >
  init(Reducer<OP, T, VType>& red) {
    red.val.val = OP<T,T,T>::identity();
  }

  // Combine
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T, typename VType>
  camp::concepts::enable_if< std::is_same< EXEC_POL, RAJA::seq_exec> >
  combine(Reducer<OP, T, VType>& out, const Reducer<OP, T, VType>& in) {
    out.val.val = OP<T,T,T>{}(out.val.val, in.val.val);
  }

  // Resolve
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T, typename VType>
  camp::concepts::enable_if< std::is_same< EXEC_POL, RAJA::seq_exec> >
  resolve(Reducer<OP, T, VType>& red) {
    *red.target = OP<T,T,T>{}(*red.target, red.val.val);
  }

} //  namespace detail
} //  namespace expt
} //  namespace RAJA

#endif //  NEW_REDUCE_SEQ_REDUCE_HPP
