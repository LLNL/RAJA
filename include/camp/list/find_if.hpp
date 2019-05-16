//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef CAMP_LIST_FIND_IF_HPP
#define CAMP_LIST_FIND_IF_HPP

#include <cstddef>
#include <type_traits>

#include "camp/lambda.hpp"
#include "camp/list/list.hpp"
#include "camp/number.hpp"
#include "camp/value.hpp"

namespace camp
{

/// \cond
namespace detail
{
  template <template <typename...> class Cond, typename... Elements>
  struct _find_if;
  template <template <typename...> class Cond, typename First, typename... Rest>
  struct _find_if<Cond, First, Rest...> {
    using type = if_<typename Cond<First>::type,
                     First,
                     typename _find_if<Cond, Rest...>::type>;
  };
  template <template <typename...> class Cond>
  struct _find_if<Cond> {
    using type = nil;
  };
}  // namespace detail
/// \endcond

template <template <typename...> class Cond, typename Seq>
struct find_if;

// TODO: document
template <template <typename...> class Cond, typename... Elements>
struct find_if<Cond, list<Elements...>> {
  using type = typename detail::_find_if<Cond, Elements...>::type;
};

CAMP_MAKE_L(find_if);

#if defined(CAMP_TEST)
#include "camp/lambda.hpp"
namespace test
{
  template <typename Index, typename ForPol>
  struct index_matches {
    using type = typename std::is_same<Index, typename ForPol::index>::type;
  };
  template <typename Index, typename T>
  struct For {
    using index = Index;
    constexpr static std::size_t value = Index::value;
  };
  CHECK_TSAME((find_if<std::is_pointer, list<float, double, int*>>), (int*));
  CHECK_TSAME((find_if<std::is_pointer, list<float, double>>), (nil));
  CHECK_TSAME((find_if_l<bind_front<std::is_same, For<num<1>, int>>,
                         list<For<num<0>, int>, For<num<1>, int>>>),
              (For<num<1>, int>));
  CHECK_TSAME((find_if_l<bind_front<index_matches, num<1>>,
                         list<For<num<0>, int>, For<num<1>, int>>>),
              (For<num<1>, int>));
}  // namespace test
#endif

}  // end namespace camp

#endif /* CAMP_LIST_FIND_IF_HPP */
