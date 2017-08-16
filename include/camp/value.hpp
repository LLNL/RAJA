#ifndef __CAMP_value_hpp
#define __CAMP_value_hpp

namespace camp
{
namespace detail
{
  struct nil;
}
// TODO: document
template <typename val = detail::nil>
struct value {
  using type = val;
};
}

#endif /* __CAMP_value_hpp */

