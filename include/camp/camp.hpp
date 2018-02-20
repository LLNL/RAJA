#ifndef __CAMP_HPP
#define __CAMP_HPP

#include <array>
#include <type_traits>

#include "camp/defines.hpp"
#include "camp/helpers.hpp"
#include "camp/lambda.hpp"
#include "camp/list/at.hpp"
#include "camp/list/find_if.hpp"
#include "camp/map.hpp"
#include "camp/number.hpp"
#include "camp/tuple.hpp"
#include "camp/value.hpp"

namespace camp
{
// Fwd
template <typename... Ts>
struct list;
template <typename T>
struct size;
template <typename Seq>
struct flatten;

// Sequences
//// list

template <typename Seq, typename T>
struct append;
template <typename... Elements, typename T>
struct append<list<Elements...>, T> {
  using type = list<Elements..., T>;
};

template <typename Seq, typename T>
struct prepend;
template <typename... Elements, typename T>
struct prepend<list<Elements...>, T> {
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

CAMP_MAKE_L(accumulate);

#if defined(CAMP_TEST)
namespace test
{
  CHECK_TSAME((accumulate<append, list<>, list<int, float, double>>),
              (list<int, float, double>));
}
#endif

template<typename T, typename L>
struct index_of;
template<typename T, typename ...Elements>
struct index_of<T, list<Elements...>> {
  template<typename Seq, typename Item>
  using inc_until = if_<typename std::is_same<T, Item>::type,
                        if_c<Seq::size == 1,
                             typename prepend<Seq, num<first<Seq>::value>>::type,
                             Seq>,
                        list<num<first<Seq>::value + 1>>
                               >;
  using indices = typename accumulate<inc_until, list<num<0>>, list<Elements...>>::type;
  using type = typename if_c<indices::size == 2, first<indices>, camp::nil>::type;
};

#if defined(CAMP_TEST)
namespace test
{
  CHECK_TSAME((index_of<int, list<>>), (nil));
  CHECK_TSAME((index_of<int, list<float, double, int>>), (num<2>));
  CHECK_TSAME((index_of<int, list<float, double, int, int, int, int>>), (num<2>));
  // CHECK_TSAME((find_if<std::is_pointer, list<float, double>>), (nil));
  // CHECK_TSAME((find_if_l<bind_front<std::is_same, For<num<1>, int>>,
  //                        list<For<num<0>, int>, For<num<1>, int>>>),
  //             (For<num<1>, int>));
  // CHECK_TSAME((find_if_l<bind_front<index_matches, num<1>>,
  //                        list<For<num<0>, int>, For<num<1>, int>>>),
  //             (For<num<1>, int>));
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

CAMP_MAKE_L(filter);

#if defined(CAMP_TEST)
namespace test
{
  CHECK_TSAME((filter<std::is_pointer, list<int, float*, double, short*>>),
              (list<float*, short*>));
}
#endif

namespace detail
{
  template <typename T>
  struct _as_list;
  template <template <typename...> class T, typename... Args>
  struct _as_list<T<Args...>> {
    using type = list<Args...>;
  };
  template <typename T, T... Args>
  struct _as_list<int_seq<T, Args...>> {
    using type = list<integral_constant<T, Args>...>;
  };
} /* detail */

template <typename T>
struct as_list_s : detail::_as_list<T>::type {
};

template <typename T>
using as_list = typename as_list_s<T>::type;

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
int main(int argc, char* argv[])
{
  camp::tuple<int, float> b;
  return 0;
}
#endif

#endif /* __CAMP_HPP */
