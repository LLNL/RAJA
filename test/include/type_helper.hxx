#ifndef _TYPE_HELPER_HPP_
#define _TYPE_HELPER_HPP_

#include <tuple>

namespace types
{


template <typename S, typename T>
struct type_cat;

template <typename... Ss, typename... Ts>
struct type_cat<std::tuple<Ss...>, std::tuple<Ts...>> {
  typedef std::tuple<Ss..., Ts...> type;
};


template <typename S, typename T>
struct product;

template <typename S, typename... Ss, typename... Ts>
struct product<std::tuple<S, Ss...>, std::tuple<Ts...>> {
  // the cartesian product of {S} and {Ts...}
  // is a list of pairs -- here: a std::tuple of 2-element std::tuples
  typedef std::tuple<std::tuple<S, Ts>...> S_cross_Ts;

  // the cartesian product of {Ss...} and {Ts...} (computed recursively)
  typedef
      typename product<std::tuple<Ss...>, std::tuple<Ts...>>::type Ss_cross_Ts;

  // concatenate both products
  typedef typename type_cat<S_cross_Ts, Ss_cross_Ts>::type type;
};

template <typename... Ss, typename... Ts, typename... Smembers>
struct product<std::tuple<std::tuple<Smembers...>, Ss...>, std::tuple<Ts...>> {
  // the cartesian product of {S} and {Ts...}
  // is a list of pairs -- here: a std::tuple of 2-element std::tuples
  typedef std::tuple<std::tuple<Smembers..., Ts>...> S_cross_Ts;

  // the cartesian product of {Ss...} and {Ts...} (computed recursively)
  typedef
      typename product<std::tuple<Ss...>, std::tuple<Ts...>>::type Ss_cross_Ts;

  // concatenate both products
  typedef typename type_cat<S_cross_Ts, Ss_cross_Ts>::type type;
};

// end the recursion
template <typename... Ts>
struct product<std::tuple<>, std::tuple<Ts...>> {
  typedef std::tuple<> type;
};
}

#endif
