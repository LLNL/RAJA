#ifndef NEW_REDUCE_SEQ_REDUCE_HPP
#define NEW_REDUCE_SEQ_REDUCE_HPP

#include "RAJA/pattern/params/reducer.hpp"

namespace RAJA {
namespace expt {
namespace detail {

  // Init
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T, typename I, typename VType>
  camp::concepts::enable_if< std::is_same< EXEC_POL, RAJA::seq_exec> >
  init(Reducer<OP, T, I, VType>& red) {
    using VT = typename Reducer<OP, T, I, VType>::value_type;
    red.valop_m.val = OP<VT,VT,VT>::identity();
  }

  // Combine
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T, typename I, typename VType>
  camp::concepts::enable_if< std::is_same< EXEC_POL, RAJA::seq_exec> >
  combine(Reducer<OP, T, I, VType>& out, const Reducer<OP, T, I, VType>& in) {
    using VT = typename Reducer<OP, T, I, VType>::value_type;
    out.valop_m.val = OP<VT,VT,VT>{}(out.valop_m.val, in.valop_m.val);
  }

  // Resolve
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T, typename I, typename VType>
  camp::concepts::enable_if< std::is_same< EXEC_POL, RAJA::seq_exec> >
  resolve(Reducer<OP, T, I, VType>& red) {
    using VT = typename Reducer<OP, T, I, VType>::value_type;
    red.set(OP<VT,VT,VT>{}(*red.target, red.valop_m.val));
  }

} //  namespace detail
} //  namespace expt
} //  namespace RAJA

#endif //  NEW_REDUCE_SEQ_REDUCE_HPP
