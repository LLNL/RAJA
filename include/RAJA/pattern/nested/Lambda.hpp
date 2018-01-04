#ifndef RAJA_pattern_nested_Lambda_HPP
#define RAJA_pattern_nested_Lambda_HPP


#include "RAJA/config.hpp"
#include "RAJA/policy/cuda.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/nested/internal.hpp"

#include "RAJA/util/chai_support.hpp"

#include "camp/camp.hpp"
#include "camp/concepts.hpp"
#include "camp/tuple.hpp"

#include <iostream>
#include <type_traits>

namespace RAJA
{

namespace nested
{


/*!
 * A nested::forall statement that executes a lambda function.
 *
 * The lambda is specified by it's index, which is defined by the order in
 * which it was specified in the call to nested::forall.
 *
 * for example:
 * RAJA::nested::forall(pol{}, make_tuple{s0, s1, s2}, lambda0, lambda1);
 *
 */
template <camp::idx_t BodyIdx>
struct Lambda : internal::Statement<camp::nil> {
  const static camp::idx_t loop_body_index = BodyIdx;
};



namespace internal{

template <camp::idx_t LoopIndex>
struct StatementExecutor<Lambda<LoopIndex>>{

  template <typename WrappedBody>
  RAJA_INLINE
  void operator()(Lambda<LoopIndex> const &, WrappedBody const &wrap)
  {
    invoke_lambda<LoopIndex>(wrap.data);
  }
};

} // namespace internal

}  // end namespace nested
}  // end namespace RAJA



#endif /* RAJA_pattern_nested_HPP */
