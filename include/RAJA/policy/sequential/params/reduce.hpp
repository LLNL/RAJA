#ifndef NEW_REDUCE_SEQ_REDUCE_HPP
#define NEW_REDUCE_SEQ_REDUCE_HPP

#include "RAJA/pattern/params/reducer.hpp"

namespace RAJA {
namespace expt {
namespace detail {

  // Init
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T>
  camp::concepts::enable_if< std::is_same< EXEC_POL, RAJA::seq_exec> >
  init(Reducer<OP<ValOp<ValLoc<T>,OP>, ValOp<ValLoc<T>,OP>, ValOp<ValLoc<T>,OP>>, ValOp<ValLoc<T>,OP>>& red) {
    red.val = OP<ValOp<ValLoc<T>,OP>, ValOp<ValLoc<T>,OP>, ValOp<ValLoc<T>,OP>>::identity();
  }
  // Combine
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T>
  camp::concepts::enable_if< std::is_same< EXEC_POL, RAJA::seq_exec> >
  combine(Reducer<OP<ValOp<ValLoc<T>,OP>, ValOp<ValLoc<T>,OP>, ValOp<ValLoc<T>,OP>>, ValOp<ValLoc<T>,OP>>& out, const Reducer<OP<ValOp<ValLoc<T>,OP>, ValOp<ValLoc<T>,OP>, ValOp<ValLoc<T>,OP>>, ValOp<ValLoc<T>,OP>>& in) {
    out.val = OP<ValOp<ValLoc<T>,OP>, ValOp<ValLoc<T>,OP>, ValOp<ValLoc<T>,OP>>{}(out.val, in.val);
  }
  // Resolve
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T>
  camp::concepts::enable_if< std::is_same< EXEC_POL, RAJA::seq_exec> >
  resolve(Reducer<OP<ValOp<ValLoc<T>,OP>, ValOp<ValLoc<T>,OP>, ValOp<ValLoc<T>,OP>>, ValOp<ValLoc<T>,OP>>& red) {
    *red.target = OP<ValOp<ValLoc<T>,OP>, ValOp<ValLoc<T>,OP>, ValOp<ValLoc<T>,OP>>{}(*red.target, red.val);
  }

  // Init
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T>
  camp::concepts::enable_if< std::is_same< EXEC_POL, RAJA::seq_exec> >
  init(Reducer<OP<ValOp<T,OP>, ValOp<T,OP>, ValOp<T,OP>>, ValOp<T,OP>>& red) {
    // RCC trying to understand what these types are
    //decltype(red.val)::nothing;
    //decltype(OP<T,T,T>::identity())::nothing;
    //red.val comes from include/RAJA/pattern/params/reducer.hpp struct Reducer{value_type val = op::identity()}
    // RCC doesn't work red.val = ValOp<T,OP>::identity(); // need to get OP<T,T,T>::identity()
    // CHANGE THIS TO USE A set() function for val in ValOp
    red.val.val = OP<T,T,T>::identity();
  }
  // Combine
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T>
  camp::concepts::enable_if< std::is_same< EXEC_POL, RAJA::seq_exec> >
  combine(Reducer<OP<ValOp<T,OP>, ValOp<T,OP>, ValOp<T,OP>>, ValOp<T,OP>>& out, const Reducer<OP<ValOp<T,OP>, ValOp<T,OP>, ValOp<T,OP>>, ValOp<T,OP>>& in) {
    out.val.val = OP<T,T,T>{}(out.val.val, in.val.val);
  }
  // Resolve
  template<typename EXEC_POL, template <typename, typename, typename> class OP, typename T>
  camp::concepts::enable_if< std::is_same< EXEC_POL, RAJA::seq_exec> >
  resolve(Reducer<OP<ValOp<T,OP>, ValOp<T,OP>, ValOp<T,OP>>, ValOp<T,OP>>& red) {
    red.target->val = OP<T,T,T>{}(red.target->val, red.val.val);
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
