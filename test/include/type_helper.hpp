/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file for type helpers used in tests.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef _TYPE_HELPER_HPP_
#define _TYPE_HELPER_HPP_

#include "gtest/gtest.h"

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


namespace tt
{
template <typename...>
struct concat;

template <template <class...> class T, typename U>
struct concat<T<U>> {
  using type = U;
};

template <typename T>
struct concat<T> {
  using type = T;
};

template <template <class...> class T,
          class... Front,
          class... Next,
          class... Rest>
struct concat<T<Front...>, T<Next...>, Rest...> {
  using type = typename concat<T<Front..., Next...>, Rest...>::type;
};

template <typename... Ts>
using concat_t = typename concat<Ts...>::type;

template <class T>
struct collapse {
  using type = T;
};

template <template <class...> class T, class... U>
struct collapse<T<T<U...>>> {
  using type = typename collapse<T<U...>>::type;
};

template <typename T>
using collapse_t = typename collapse<T>::type;

template <template <class> class, class>
struct apply;

template <template <class...> class L, template <class> class Fn, class... Ts>
struct apply<Fn, L<Ts...>> {
  using type = collapse_t<L<concat_t<Fn<Ts>...>>>;
};

template <template <class> class Outer, class T>
using apply_t = typename apply<Outer, T>::type;

} // closing brace for namespace tt


namespace detail
{
template <typename T>
struct ForTesting;

template <template <class...> class T, typename... Ts>
struct ForTesting<T<Ts...>> {
  using type = ::testing::Types<Ts...>;
};
}

template <typename T>
using ForTesting = typename ::detail::ForTesting<T>::type;

#endif
