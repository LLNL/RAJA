//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef CAMP_NUMBER_IF_HPP
#define CAMP_NUMBER_IF_HPP

#include "camp/value.hpp"

#include <type_traits>

namespace camp
{

// TODO: document
template <bool Cond,
          typename Then = camp::true_type,
          typename Else = camp::false_type>
struct if_cs {
  using type = Then;
};

template <typename Then, typename Else>
struct if_cs<false, Then, Else> {
  using type = Else;
};

// TODO: document
template <bool Cond,
          typename Then = camp::true_type,
          typename Else = camp::false_type>
using if_c = typename if_cs<Cond, Then, Else>::type;

// TODO: document
template <typename Cond,
          typename Then = camp::true_type,
          typename Else = camp::false_type>
struct if_s : if_cs<Cond::value, Then, Else> {
};

template <typename Then, typename Else>
struct if_s<nil, Then, Else> : if_cs<false, Then, Else> {
};

// TODO: document
template <typename... Ts>
using if_ = typename if_s<Ts...>::type;

}  // end namespace camp

#endif /* CAMP_NUMBER_IF_HPP */
