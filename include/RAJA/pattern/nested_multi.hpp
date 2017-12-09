#ifndef RAJA_pattern_nested_multi_HPP
#define RAJA_pattern_nested_multi_HPP


#include "RAJA/pattern/nested.hpp"


namespace RAJA
{
namespace nested
{



template<typename NestedPolicy, typename SegmentTuple, typename Body>
RAJA_INLINE
auto makeLoop(NestedPolicy const &p, SegmentTuple const &s, Body const &b) ->
  LoopData<NestedPolicy, SegmentTuple, Body>
{
  return LoopData<NestedPolicy, SegmentTuple, Body>{p,s,b};
}



template <typename UberPolicy, typename ... LoopList>
RAJA_INLINE void forall_multi(LoopList && ... loops)
{
  forall_multi(UberPolicy{}, std::forward<LoopList>(loops)...);
}




}  // end namespace nested
}  // end namespace RAJA



#endif /* RAJA_pattern_nested_HPP */
