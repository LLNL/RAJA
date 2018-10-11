#ifndef __CAMP_HPP
#define __CAMP_HPP

#include <array>
#include <type_traits>

#include "camp/defines.hpp"
#include "camp/helpers.hpp"
#include "camp/lambda.hpp"
#include "camp/list.hpp"
#include "camp/map.hpp"
#include "camp/number.hpp"
#include "camp/size.hpp"
#include "camp/tuple.hpp"
#include "camp/value.hpp"

namespace camp
{
// Fwd
template <typename... Ts>
struct list;
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
}  // namespace detail

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
}  // namespace test
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
}  // namespace test
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
}  // namespace detail

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


/**
 * @brief Get the index of the first instance of T in L
 */
template <typename T, typename L>
struct index_of;
template <typename T, typename... Elements>
struct index_of<T, list<Elements...>> {
  template <typename Seq, typename Item>
  using inc_until =
      if_<typename std::is_same<T, Item>::type,
          if_c<size<Seq>::value == 1,
               typename prepend<Seq, num<first<Seq>::value>>::type,
               Seq>,
          list<num<first<Seq>::value + 1>>>;
  using indices =
      typename accumulate<inc_until, list<num<0>>, list<Elements...>>::type;
  using type =
      typename if_c<size<indices>::value == 2, first<indices>, camp::nil>::type;
};

#if defined(CAMP_TEST)
namespace test
{
  CHECK_TSAME((index_of<int, list<>>), (nil));
  CHECK_TSAME((index_of<int, list<float, double, int>>), (num<2>));
  CHECK_TSAME((index_of<int, list<float, double, int, int, int, int>>),
              (num<2>));
  // CHECK_TSAME((find_if<std::is_pointer, list<float, double>>), (nil));
  // CHECK_TSAME((find_if_l<bind_front<std::is_same, For<num<1>, int>>,
  //                        list<For<num<0>, int>, For<num<1>, int>>>),
  //             (For<num<1>, int>));
  // CHECK_TSAME((find_if_l<bind_front<index_matches, num<1>>,
  //                        list<For<num<0>, int>, For<num<1>, int>>>),
  //             (For<num<1>, int>));
}  // namespace test
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

//// size

#if defined(CAMP_TEST)
namespace test
{
  CHECK_IEQ((size<list<int>>), (1));
  CHECK_IEQ((size<list<int, int>>), (2));
  CHECK_IEQ((size<list<int, int, int>>), (3));
}  // namespace test
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
}  // namespace test
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
