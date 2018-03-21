#ifndef CAMP_LIST_LIST_HPP
#define CAMP_LIST_LIST_HPP

#include "camp/number/number.hpp"

namespace camp
{
// TODO: document

template <typename... Ts>
struct list {
  using type = list;
};

template <typename T>
struct as_list_s;

template <template <typename...> class T, typename... Args>
struct as_list_s<T<Args...>> {
  using type = list<Args...>;
};

template <typename T>
using as_list = typename as_list_s<T>::type;

}  // namespace camp

#endif /* CAMP_LIST_LIST_HPP */
