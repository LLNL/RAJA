/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA-specific type traits
 *
 *          Definitions in this file will propagate to all RAJA header files.
 *
 ******************************************************************************
 */

#ifndef RAJA_metalib_HPP
#define RAJA_metalib_HPP

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <type_traits>

namespace RAJA {

// Adopted from range-v3 and Eric Niebler's meta library
// Only extracted relevent parts
namespace metalib {

template <typename...>
struct list;

template <bool B>
using bool_ = std::integral_constant<bool, B>;

namespace impl {
#ifdef __clang__
// Clang is faster with this implementation
template <typename, typename = bool>
struct _if_ {};

template <typename If>
struct _if_<list<If>, decltype(bool(If::type::value))>
    : std::enable_if<If::type::value> {};

template <typename If, typename Then>
struct _if_<list<If, Then>, decltype(bool(If::type::value))>
    : std::enable_if<If::type::value, Then> {};

template <typename If, typename Then, typename Else>
struct _if_<list<If, Then, Else>, decltype(bool(If::type::value))>
    : std::conditional<If::type::value, Then, Else> {};
#else
// GCC seems to prefer this implementation
template <typename, typename = std::true_type>
struct _if_ {};

template <typename If>
struct _if_<list<If>, bool_<If::type::value>> {
  using type = void;
};

template <typename If, typename Then>
struct _if_<list<If, Then>, bool_<If::type::value>> {
  using type = Then;
};

template <typename If, typename Then, typename Else>
struct _if_<list<If, Then, Else>, bool_<If::type::value>> {
  using type = Then;
};

template <typename If, typename Then, typename Else>
struct _if_<list<If, Then, Else>, bool_<!If::type::value>> {
  using type = Else;
};
#endif
} // namespace detail

template <typename... Ts>
using if_ = typename impl::_if_<list<Ts...>>::type;

template <bool If, typename... Args>
using if_c = typename impl::_if_<list<bool_<If>, Args...>>::type;

// concept checking functions

/// metafunction for use within decltype expression to validate aribitrary expressions
template <typename... T>
constexpr std::true_type
valid_expr(T &&...);

/// helper metafunction for use within decltype expression for validation
template <typename Ret, typename T>
constexpr Ret
returns(T const &);

/// metafunction for use within decltype expression to validate return type is convertible to given type
template <typename T, typename U>
constexpr auto
convertible_to(U &&u)
    -> decltype(returns<std::true_type>(static_cast<T>((U &&) u)));

/// metafunction for use within decltype expression to validate type of expression
template <typename T, typename U>
constexpr auto has_type(U &&) -> if_<std::is_same<T, U>, std::true_type>;

/// bool list -- use for {all,none,any}_of metafunctions
template <bool...>
struct blist;

/// negation metafunction of a value type
template <typename T>
using negate_t = bool_<!T::value>;

/// all_of metafunction of a value type list -- all must be "true"
template <bool ... Bs>
using all_of = std::is_same<blist<true, Bs...>, blist<Bs..., true>>;

/// none_of metafunction of a value type list -- all must be "false"
template <bool ... Bs>
using none_of = std::is_same<blist<false, Bs...>, blist<Bs..., false>>;

/// any_of metafunction of a value type list -- at least one must be "true""
template <bool ... Bs>
using any_of = negate_t<none_of<Bs...>>;

/// all_of metafunction of a bool list -- all must be "true"
template <typename... Bs>
using all_of_t = all_of<Bs::value...>;

/// none_of metafunction of a bool list -- all must be "false"
template <typename ... Bs>
using none_of_t = none_of<Bs::value...>;

/// any_of metafunction of a bool list -- at least one must be "true""
template <typename ... Bs>
using any_of_t = any_of<Bs::value...>;

/// helper metafunction for use within `valid_expr` for concept requirement aggregation
template <typename T>
auto
models() -> if_<std::is_same<std::true_type, T>, std::true_type>;

} // end namespace meta

} // end namespace RAJA

#endif
