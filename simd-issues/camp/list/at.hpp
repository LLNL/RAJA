#ifndef __CAMP_list_at_hpp
#define __CAMP_list_at_hpp

#include "camp/defines.hpp"
#include "camp/helpers.hpp"
#include "camp/list/list.hpp"
#include "camp/number.hpp"
#include "camp/value.hpp"

namespace camp
{

namespace detail
{
  // Lookup from metal::at machinery
  template <idx_t, typename>
  struct entry {
  };

  template <typename, typename>
  struct entries;

  template <idx_t... keys, typename... vals>
  struct entries<idx_seq<keys...>, list<vals...>> : entry<keys, vals>... {
  };

  template <idx_t key, typename val>
  value<val> _lookup_impl(entry<key, val>*);

  template <typename>
  value<> _lookup_impl(...);

  template <typename vals, typename indices, idx_t Idx>
  struct _lookup
      : decltype(_lookup_impl<Idx>(declptr<entries<indices, vals>>())) {
  };

  template <typename T, idx_t Idx>
  struct _at;
  template <template <class...> class T, typename X, typename... Rest>
  struct _at<T<X, Rest...>, 0> {
    using type = X;
  };
  template <template <class...> class T,
            typename X,
            typename Y,
            typename... Rest>
  struct _at<T<X, Y, Rest...>, 1> {
    using type = Y;
  };
  template <template <class...> class T, idx_t Idx, typename... Rest>
  struct _at<T<Rest...>, Idx> {
    using type = typename _lookup<T<Rest...>,
                                  make_idx_seq_t<sizeof...(Rest)>,
                                  Idx>::type;
  };
}

// TODO: document
template <typename Seq, typename Num>
struct at;
template <typename T, idx_t Val>
struct at<T, num<Val>> {
  using type = typename detail::_at<T, Val>::type;
};


template <typename T>
using first = typename at<T, num<0>>::type;

template <typename T>
using second = typename at<T, num<1>>::type;

// TODO: document
template <typename T, idx_t Idx>
using at_v = typename at<T, num<Idx>>::type;

// TODO: document
template <typename T, typename U>
using at_t = typename at<T, U>::type;
}


#endif /* __CAMP_list_at_hpp */
