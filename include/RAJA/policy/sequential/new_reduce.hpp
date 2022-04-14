#ifndef NEW_REDUCE_SEQ_REDUCE_HPP
#define NEW_REDUCE_SEQ_REDUCE_HPP

#include "RAJA/pattern/params/new_reduce_basic.hpp"

namespace RAJA {
namespace expt {
namespace detail {

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
    *red.target = OP{}(red.val, *red.target);
  }

} //  namespace detail
} //  namespace expt
} //  namespace RAJA

#endif //  NEW_REDUCE_SEQ_REDUCE_HPP
