#ifndef CAMP_LIST_LIST_HPP
#define CAMP_LIST_LIST_HPP

#include "camp/number/number.hpp"

namespace camp
{
// TODO: document

template <typename... Ts>
struct list {
  using type = list;
  static constexpr idx_t size = sizeof...(Ts);
};
}

#endif /* CAMP_LIST_LIST_HPP */
