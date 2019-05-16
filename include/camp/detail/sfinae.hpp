//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef CAMP_DETAIL_SFINAE_HPP
#define CAMP_DETAIL_SFINAE_HPP

#include "camp/helpers.hpp"
#include "camp/number/number.hpp"
#include "camp/value.hpp"

#include <type_traits>

namespace camp
{

/// \cond
namespace detail
{

  // caller pattern from metal library
  template <template <typename...> class expr, typename... vals>
  struct caller;

  template <template <typename...> class expr,
            typename... vals,
            typename std::enable_if<is_value<expr<vals...>>::value>::type* =
                nullptr>
  value<expr<vals...>> sfinae(caller<expr, vals...>*);

  value<> sfinae(...);

  template <template <typename...> class expr, typename... vals>
  struct caller : decltype(sfinae(declptr<caller<expr, vals...>>())) {
  };

  template <template <typename...> class Expr, typename... Vals>
  struct call_s : caller<Expr, Vals...> {
  };

  template <template <typename...> class Expr, typename... Vals>
  using call = Expr<Vals...>;
};  // namespace detail
/// \endcond

}  // end namespace camp

#endif /* CAMP_DETAIL_SFINAE_HPP */
