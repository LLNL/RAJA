/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA concept definitions.
 *
 *          Definitions in this file will propagate to all RAJA header files.
 *
 ******************************************************************************
 */

#ifndef RAJA_concepts_HPP
#define RAJA_concepts_HPP

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

#include <iterator>
#include <type_traits>

namespace RAJA
{

namespace concepts
{

template <typename T>
using decay = typename std::remove_reference<typename std::remove_cv<typename std::decay<T>::type>::type>::type;

namespace metalib
{

template <typename T, T Val>
struct integral_constant {
  using type = T;
  static constexpr T value = Val;
};

template <bool B>
using bool_ = integral_constant<bool, B>;
template <int I>
using int_ = integral_constant<int, I>;

using true_type = bool_<true>;
using false_type = bool_<false>;

template <typename...>
struct list;

namespace impl
{

#ifdef __clang__

// Clang is faster with this implementation
template <typename, typename = bool>
struct _if_ {
};

template <typename If>
struct _if_<list<If>, decltype(bool(If::type::value))>
    : std::enable_if<If::type::value> {
};

template <typename If, typename Then>
struct _if_<list<If, Then>, decltype(bool(If::type::value))>
    : std::enable_if<If::type::value, Then> {
};

template <typename If, typename Then, typename Else>
struct _if_<list<If, Then, Else>, decltype(bool(If::type::value))>
    : std::conditional<If::type::value, Then, Else> {
};

#else

// GCC seems to prefer this implementation
template <typename, typename = true_type>
struct _if_ {
};

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

}  // namespace detail

template <typename... Ts>
using if_ = typename impl::_if_<list<Ts...>>::type;

template <bool If, typename... Args>
using if_c = typename impl::_if_<list<bool_<If>, Args...>>::type;

/// bool list -- use for {all,none,any}_of metafunctions
template <bool...>
struct blist;

/// negation metafunction of a value type
template <typename T>
using negate_t = bool_<!T::value>;

/// all_of metafunction of a value type list -- all must be "true"
template <bool... Bs>
using all_of = std::is_same<blist<true, Bs...>, blist<Bs..., true>>;

/// none_of metafunction of a value type list -- all must be "false"
template <bool... Bs>
using none_of = std::is_same<blist<false, Bs...>, blist<Bs..., false>>;

/// any_of metafunction of a value type list -- at least one must be "true""
template <bool... Bs>
using any_of = negate_t<none_of<Bs...>>;

/// all_of metafunction of a bool list -- all must be "true"
template <typename... Bs>
using all_of_t = all_of<Bs::value...>;

/// none_of metafunction of a bool list -- all must be "false"
template <typename... Bs>
using none_of_t = none_of<Bs::value...>;

/// any_of metafunction of a bool list -- at least one must be "true""
template <typename... Bs>
using any_of_t = any_of<Bs::value...>;

}  // end namespace metalib

namespace detail
{
template <class...>
struct voider {
  using type = void;
};

template <class... T>
using void_t = typename voider<T...>::type;

template <class, template <class...> class Op, class... Args>
struct detector {
  using value_t = metalib::false_type;
};

template <template <class...> class Op, class... Args>
struct detector<void_t<Op<Args...>>, Op, Args...> {
  using value_t = metalib::true_type;
  using type = metalib::true_type;
};

template <template <class...> class Op, class... Args>
using detected = typename detector<void, Op, Args...>::type;
} // end brace namespace detail

template <template <class...> class Op, class... Args>
struct requires_ : detail::detector<void, Op, Args...>::value_t{};

  template <typename T>
  using negate = metalib::negate_t<T>;

/// metafunction to get instance of value type for concepts
template <typename T>
T &&val() noexcept;

/// metafunction to get instance of const type for concepts
template <typename T>
T const &&cval() noexcept;

template <typename Ret, typename T>
Ret returns(T const &) noexcept;

/// metafunction for use within decltype expression to validate return type is
/// convertible to given type
template <typename T, typename U>
constexpr auto convertible_to(U &&u) noexcept
    -> decltype(returns<metalib::true_type>(static_cast<T>((U &&) u)));

/// metafunction for use within decltype expression to validate type of
/// expression
template <typename T, typename U>
constexpr auto has_type(U &&) noexcept
    -> metalib::if_<std::is_same<T, U>, metalib::true_type>;

template <template <class...> class Concept, class... Ts>
constexpr detail::detected<Concept, Ts...> models() noexcept;

template <typename BoolLike>
constexpr auto conforms(BoolLike) noexcept
    -> metalib::if_<BoolLike, metalib::true_type>;

/// metaprogramming concept for SFINAE checking of aggregating concepts
template <typename... Args>
using all_of = metalib::all_of_t<Args...>;

template <typename... Args>
using none_of = metalib::none_of_t<Args...>;

template <typename... Args>
using any_of = metalib::any_of_t<Args...>;

template <typename... Args>
using enable_if = typename std::enable_if<all_of<Args...>::value>::type;

using concepts::metalib::bool_;

}  // end namespace concepts

}  // end namespace RAJA


template <typename... T>
RAJA::concepts::metalib::true_type ___valid_expr___(T &&...) noexcept;
#define DefineConcept(...) decltype(___valid_expr___(__VA_ARGS__))

#define DefineTypeTraitFromConcept(TTName, ConceptName) \
template <typename ... Args> \
using TTName = RAJA::concepts::requires_<ConceptName, Args...>

namespace ___hidden_concepts
{

using namespace RAJA::concepts;

template <typename T>
using LessThanComparable = DefineConcept(convertible_to<bool>(val<T>()
                                                              < val<T>()));

template <typename T>
using GreaterThanComparable = DefineConcept(convertible_to<bool>(val<T>()
                                                                 > val<T>()));

template <typename T>
using LessEqualComparable = DefineConcept(convertible_to<bool>(val<T>()
                                                               <= val<T>()));

template <typename T>
using GreaterEqualComparable = DefineConcept(convertible_to<bool>(val<T>()
                                                                  >= val<T>()));

template <typename T>
using EqualityComparable = DefineConcept(convertible_to<bool>(val<T>()
                                                              == val<T>()));

template <typename T, typename U>
using ComparableTo = DefineConcept(convertible_to<bool>(val<U>() < val<T>()),
                                   convertible_to<bool>(val<T>() < val<U>()),
                                   convertible_to<bool>(val<U>() <= val<T>()),
                                   convertible_to<bool>(val<T>() <= val<U>()),
                                   convertible_to<bool>(val<U>() > val<T>()),
                                   convertible_to<bool>(val<T>() > val<U>()),
                                   convertible_to<bool>(val<U>() >= val<T>()),
                                   convertible_to<bool>(val<T>() >= val<U>()),
                                   convertible_to<bool>(val<U>() == val<T>()),
                                   convertible_to<bool>(val<T>() == val<U>()),
                                   convertible_to<bool>(val<U>() != val<T>()),
                                   convertible_to<bool>(val<T>() != val<U>()));

template <typename T>
using Comparable = ComparableTo<T, T>;

template <typename T>
using plain_t = typename std::remove_reference<T>::type;

template <typename T>
using diff_t = decltype(val<plain_t<T>>() - val<plain_t<T>>());

template <typename T>
using iterator_t = decltype(std::begin(val<plain_t<T>>()));

template <typename T>
using Iterator = DefineConcept(*val<T>(), has_type<T &>(++val<T &>()));

template <typename T>
using ForwardIterator = DefineConcept(Iterator<T>(),
                                      val<T &>()++,
                                      *val<T &>()++);

template <typename T>
using BidirectionalIterator =
    DefineConcept(ForwardIterator<T>(),
                  has_type<T &>(--val<T &>()),
                  convertible_to<T const &>(val<T &>()--),
                  *val<T &>()--);

template <typename T>
using RandomAccessIterator =
    DefineConcept(BidirectionalIterator<T>(),
                  Comparable<T>(),
                  has_type<T &>(val<T &>() += val<diff_t<T>>()),
                  has_type<T>(val<T>() + val<diff_t<T>>()),
                  has_type<T>(val<diff_t<T>>() + val<T>()),
                  has_type<T &>(val<T &>() -= val<diff_t<T>>()),
                  has_type<T>(val<T>() - val<diff_t<T>>()),
                  val<T>()[val<diff_t<T>>()]);

template <typename T>
using HasBeginEnd = DefineConcept(std::begin(val<T>()), std::end(val<T>()));

template <typename T>
using Range = DefineConcept(HasBeginEnd<T>(), Iterator<iterator_t<T>>());

template <typename T>
using ForwardRange = DefineConcept(HasBeginEnd<T>(),
                                   ForwardIterator<iterator_t<T>>());

template <typename T>
using BidirectionalRange =
    DefineConcept(HasBeginEnd<T>(), BidirectionalIterator<iterator_t<T>>());

template <typename T>
using RandomAccessRange = DefineConcept(HasBeginEnd<T>(),
                                        RandomAccessIterator<iterator_t<T>>());

template <typename T>
using Integral = DefineConcept(conforms(std::is_integral<T>()));

template <typename T>
using Signed = DefineConcept(Integral<T>(), conforms(std::is_signed<T>()));
template <typename T>
using Unsigned = DefineConcept(Integral<T>(), conforms(std::is_unsigned<T>()));

}  // end namespace ___hidden_concepts

namespace RAJA
{
namespace concepts
{

using ___hidden_concepts::EqualityComparable;
using ___hidden_concepts::LessThanComparable;
using ___hidden_concepts::GreaterThanComparable;
using ___hidden_concepts::LessEqualComparable;
using ___hidden_concepts::GreaterEqualComparable;

using ___hidden_concepts::Comparable;
using ___hidden_concepts::ComparableTo;

using ___hidden_concepts::Iterator;
using ___hidden_concepts::ForwardIterator;
using ___hidden_concepts::BidirectionalIterator;
using ___hidden_concepts::RandomAccessIterator;

using ___hidden_concepts::Range;
using ___hidden_concepts::ForwardRange;
using ___hidden_concepts::BidirectionalRange;
using ___hidden_concepts::RandomAccessRange;

using ___hidden_concepts::Integral;
using ___hidden_concepts::Signed;
using ___hidden_concepts::Unsigned;

}  // end namespace concepts
}  // end namespace RAJA

#endif
