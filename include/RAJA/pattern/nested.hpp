#ifndef RAJA_pattern_nested_HPP
#define RAJA_pattern_nested_HPP


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
 * A RAJA::nested::forall execution policy.
 *
 * This is just a list of nested::forall statements.
 */
template <typename... Stmts>
using Policy = internal::StatementList<Stmts...>;



} // namespace nested


#ifdef RAJA_ENABLE_CHAI

namespace detail
{


/*
 * Define CHAI support for nested policies.
 *
 * We need to walk the entire set of execution policies inside of the
 * RAJA::nested::Policy
 */
template <typename... POLICIES>
struct get_space<RAJA::nested::Policy<POLICIES...>>
    : public get_space_from_list<  // combines exec policies to find exec space

          // Extract just the execution policies from the tuple
          RAJA::nested::internal::get_space_policies<
              typename camp::tuple<POLICIES...>::TList>

          > {
};

}  // end detail namespace

#endif  // RAJA_ENABLE_CHAI


namespace nested
{



namespace internal{


template <camp::idx_t Index, typename BaseWrapper>
struct GenericWrapper {
  using data_type = camp::decay<typename BaseWrapper::data_type>;

  BaseWrapper wrapper;

  GenericWrapper(BaseWrapper const &w) : wrapper{w} {}
  GenericWrapper(data_type &d) : wrapper{d} {}

};


} // namespace internal



template <typename PolicyType, typename SegmentTuple, typename ... Bodies>
RAJA_INLINE void forall(const PolicyType &policy, const SegmentTuple &segments, const Bodies & ... bodies)
{
  detail::setChaiExecutionSpace<PolicyType>();

  // TODO: test that all policy members model the Executor policy concept
  // TODO: add a static_assert for functors which cannot be invoked with
  //       index_tuple
  // TODO: add assert that all Lambda<i> match supplied loop bodies

  // Create the LoopData object, which contains our policy object,
  // our segments, loop bodies, and the tuple of loop indices
  // it is passed through all of the nested::forall mechanics by-referenece,
  // and only copied to provide thread-private instances.
  auto loop_data = internal::LoopData<PolicyType, SegmentTuple, Bodies...>(policy, segments, bodies...);

  // Create a StatmentList wrapper to execute our policy (which is just
  // a StatementList)
  auto wrapper = internal::make_statement_list_wrapper(policy, loop_data);

  // Execute!
  wrapper();


  detail::clearChaiExecutionSpace();
}

}  // end namespace nested


}  // end namespace RAJA



#endif /* RAJA_pattern_nested_HPP */
