#ifndef _TYPE_HELPER_HPP_
#define _TYPE_HELPER_HPP_

#include <tuple>

namespace types
{

template <class T>
struct flatten;

template <template <typename...> class C, typename... FArgs>
struct flatten<C<FArgs...>> {
  template <typename Target, typename... B>
  struct append;

  template <typename... Args1, typename T>
  struct append<C<Args1...>, T> {
    using type = C<Args1..., T>;
  };

  template <typename Target, typename... Args>
  struct inner;

  template <typename Target, typename T>
  struct inner2;

  template <typename Target, typename... Args>
  struct inner2<Target, C<Args...>> {
    using type = typename inner<Target, Args...>::type;
  };

  template <typename Target, typename T>
  struct inner2 {
    using type = typename append<Target, T>::type;
  };

  template <typename Target, typename T, typename... Args>
  struct inner<Target, T, Args...>
      : inner<typename inner2<Target, T>::type, Args...> {
  };

  template <typename Target, typename T>
  struct inner<Target, T> {
    using type = typename inner2<Target, T>::type;
  };

  using type = typename inner<C<>, FArgs...>::type;
};

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
  typedef std::tuple<typename flatten<std::tuple<S, Ts>>::type...> S_cross_Ts;

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
