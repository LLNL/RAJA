#ifndef __CAMP_HPP
#define __CAMP_HPP

#include <array>
#include <type_traits>

#include "camp/defines.hpp"
#include "camp/list/at.hpp"
#include "camp/value.hpp"
#include "camp/number.hpp"
#include "camp/helpers.hpp"

namespace camp
{
// Fwd
template <typename... Ts>
struct list;
template <typename T>
struct as_list;
template <typename T>
struct as_array;
template <typename T>
struct size;
template <typename Seq>
struct flatten;
template <idx_t N>
struct num;

// helpers

// Lambda

template <template <typename...> class Expr>
struct lambda {
  template <typename... Ts>
  using expr = typename Expr<Ts...>::type;
};

template <typename Lambda, typename Seq>
struct apply;
template <typename Lambda, typename... Args>
struct apply<Lambda, list<Args...>> {
  using type = typename Lambda::template expr<Args...>::type;
};

template <typename Lambda, typename... Args>
struct invoke {
  using type = typename Lambda::template expr<Args...>::type;
};

template <idx_t n>
struct arg {
  template <typename... Ts>
  using expr = typename at<list<Ts...>, num<n - 1>>::type;
};

using _1 = arg<1>;
using _2 = arg<2>;
using _3 = arg<3>;
using _4 = arg<4>;
using _5 = arg<5>;
using _6 = arg<6>;
using _7 = arg<7>;
using _8 = arg<8>;
using _9 = arg<9>;

namespace detail
{
  template <typename T, typename... Args>
  struct get_bound_arg {
    using type = T;
  };
  template <idx_t i, typename... Args>
  struct get_bound_arg<arg<i>, Args...> {
    using type = typename arg<i>::template expr<Args...>;
  };
}

template <template <typename...> class Expr, typename... ArgBindings>
struct bind {
  using bindings = list<ArgBindings...>;
  template <typename... Ts>
  using expr = typename Expr<
      typename detail::get_bound_arg<ArgBindings, Ts...>::type...>::type;
};

#if defined(CAMP_TEST)
namespace test
{
  CHECK_TSAME((invoke<bind<list, _1, int, _2>, float, double>),
              (list<float, int, double>));
}
#endif

template <template <typename...> class Expr, typename... BoundArgs>
struct bind_front {
  template <typename... Ts>
  using expr = typename Expr<BoundArgs..., Ts...>::type;
};


// Numbers

template <typename Cond, typename Then, typename Else>
struct if_;
template <typename Then, typename Else>
struct if_<std::true_type, Then, Else> {
  using type = Then;
};
template <typename Then, typename Else>
struct if_<std::false_type, Then, Else> {
  using type = Else;
};

template <idx_t Cond, typename Then, typename Else>
struct if_v {
  using type = Then;
};
template <typename Then, typename Else>
struct if_v<0, Then, Else> {
  using type = Else;
};

// Sequences
//// list

template <typename Seq, typename T>
struct append;
template <typename... Elements, typename T>
struct append<list<Elements...>, T> {
  using type = list<Elements..., T>;
};

template <typename Seq, typename T>
struct extend;
template <typename... Elements, typename... NewElements>
struct extend<list<Elements...>, list<NewElements...>> {
  using type = list<Elements..., NewElements...>;
};

namespace detail
{
  template <typename CurSeq, size_t N, typename... Rest>
  struct flatten_impl;
  template <typename CurSeq>
  struct flatten_impl<CurSeq, 0> {
    using type = CurSeq;
  };
  template <typename... CurSeqElements,
            size_t N,
            typename First,
            typename... Rest>
  struct flatten_impl<list<CurSeqElements...>, N, First, Rest...> {
    using type = typename flatten_impl<list<CurSeqElements..., First>,
                                       N - 1,
                                       Rest...>::type;
  };
  template <typename... CurSeqElements,
            size_t N,
            typename... FirstInnerElements,
            typename... Rest>
  struct flatten_impl<list<CurSeqElements...>,
                      N,
                      list<FirstInnerElements...>,
                      Rest...> {
    using first_inner_flat =
        typename flatten_impl<list<>,
                              sizeof...(FirstInnerElements),
                              FirstInnerElements...>::type;
    using cur_and_first =
        typename extend<list<CurSeqElements...>, first_inner_flat>::type;
    using type = typename flatten_impl<cur_and_first, N - 1, Rest...>::type;
  };
}

template <typename... Elements>
struct flatten<list<Elements...>>
    : detail::flatten_impl<list<>, sizeof...(Elements), Elements...> {
};

#if defined(CAMP_TEST)
namespace test
{
  CHECK_TSAME((flatten<list<>>), (list<>));
  CHECK_TSAME((flatten<list<int>>), (list<int>));
  CHECK_TSAME((flatten<list<list<int>>>), (list<int>));
  CHECK_TSAME((flatten<list<list<list<int>>>>), (list<int>));
  CHECK_TSAME((flatten<list<float, list<int, double>, list<list<int>>>>),
              (list<float, int, double, int>));
}
#endif

template <template <typename...> class Op, typename T>
struct transform;
template <template <typename...> class Op, typename... Elements>
struct transform<Op, list<Elements...>> {
  using type = list<typename Op<Elements>::type...>;
};

#if defined(CAMP_TEST)
namespace test
{
  CHECK_TSAME((transform<std::add_cv, list<int>>), (list<const volatile int>));
  CHECK_TSAME((transform<std::remove_reference, list<int&, int&>>),
              (list<int, int>));
}
#endif

namespace detail
{
  template <template <typename...> class Op, typename Current, typename... Rest>
  struct accumulate_impl;
  template <template <typename...> class Op,
            typename Current,
            typename First,
            typename... Rest>
  struct accumulate_impl<Op, Current, First, Rest...> {
    using current = typename Op<Current, First>::type;
    using type = typename accumulate_impl<Op, current, Rest...>::type;
  };
  template <template <typename...> class Op, typename Current>
  struct accumulate_impl<Op, Current> {
    using type = Current;
  };
}

template <template <typename...> class Op, typename Initial, typename Seq>
struct accumulate;
template <template <typename...> class Op,
          typename Initial,
          typename... Elements>
struct accumulate<Op, Initial, list<Elements...>> {
  using type = typename detail::accumulate_impl<Op, Initial, Elements...>::type;
};

#if defined(CAMP_TEST)
namespace test
{
  CHECK_TSAME((accumulate<append, list<>, list<int, float, double>>),
              (list<int, float, double>));
}
#endif

template <template <typename...> class Op, typename Seq>
struct filter;

template <template <typename...> class Op, typename... Elements>
struct filter<Op, list<Elements...>> {
  template <typename Seq, typename T>
  using append_if =
      if_<typename Op<T>::type, typename append<Seq, T>::type, Seq>;
  using type = typename accumulate<append_if, list<>, list<Elements...>>::type;
};

#if defined(CAMP_TEST)
namespace test
{
  CHECK_TSAME((filter<std::is_pointer, list<int, float*, double, short*>>),
              (list<float*, short*>));
}
#endif

template <template <typename...> class T, typename... Args>
struct as_list<T<Args...>> {
  using type = list<Args...>;
};

template <typename T, T... Args>
struct as_list<int_seq<T, Args...>> {
  using type = list<integral<T, Args>...>;
};

//// array: TODO, using std::array for now
template <typename T, T... Vals>
struct as_array<int_seq<T, Vals...>> {
  constexpr static std::array<T, sizeof...(Vals)> value{Vals...};
};

//// size
template <typename... Args>
struct size<list<Args...>> {
  constexpr static idx_t value{sizeof...(Args)};
  using type = num<sizeof...(Args)>;
};

#if defined(CAMP_TEST)
namespace test
{
  CHECK_IEQ((size<list<int>>), (1));
  CHECK_IEQ((size<list<int, int>>), (2));
  CHECK_IEQ((size<list<int, int, int>>), (3));
}
#endif

template <typename T, T... Args>
struct size<int_seq<T, Args...>> {
  constexpr static idx_t value{sizeof...(Args)};
  using type = num<sizeof...(Args)>;
};

#if defined(CAMP_TEST)
namespace test
{
  CHECK_IEQ((size<idx_seq<0>>), (1));
  CHECK_IEQ((size<idx_seq<0, 0>>), (2));
  CHECK_IEQ((size<idx_seq<0, 0, 0>>), (3));
}
#endif

}  // end namespace camp

#if defined(CAMP_TEST)
int main(int argc, char* argv[]) { return 0; }
#endif

#endif /* __CAMP_HPP */
