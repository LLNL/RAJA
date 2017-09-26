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

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
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

#ifndef RAJA_concepts_HPP
#define RAJA_concepts_HPP

#include <iterator>
#include <type_traits>

namespace RAJA
{

namespace concepts
{

namespace metalib
{

template <class T, T v>
struct integral_constant {
  static constexpr T value = v;
  using value_type = T;
  using type = integral_constant;
  constexpr operator value_type() const noexcept { return value; }
  constexpr value_type operator()() const noexcept { return value; }
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


template <typename T, typename U>
struct is_same : false_type {
};

template <typename T>
struct is_same<T, T> : true_type {
};

/// bool list -- use for {all,none,any}_of metafunctions
template <bool...>
struct blist;

/// negation metafunction of a value type
template <typename T>
struct negate_t : bool_<!T::value> {
};

/// all_of metafunction of a value type list -- all must be "true"
template <bool... Bs>
struct all_of : metalib::is_same<blist<true, Bs...>, blist<Bs..., true>> {
};

/// none_of metafunction of a value type list -- all must be "false"
template <bool... Bs>
struct none_of : metalib::is_same<blist<false, Bs...>, blist<Bs..., false>> {
};

/// any_of metafunction of a value type list -- at least one must be "true""
template <bool... Bs>
struct any_of : negate_t<none_of<Bs...>> {
};

/// all_of metafunction of a bool list -- all must be "true"
template <typename... Bs>
struct all_of_t : all_of<Bs::value...> {
};

/// none_of metafunction of a bool list -- all must be "false"
template <typename... Bs>
struct none_of_t : none_of<Bs::value...> {
};

/// any_of metafunction of a bool list -- at least one must be "true""
template <typename... Bs>
struct any_of_t : any_of<Bs::value...> {
};

}  // end namespace metalib

}  // end namespace concepts

}  // end namespace RAJA

template <typename... T>
RAJA::concepts::metalib::true_type ___valid_expr___(T &&...) noexcept;
#define DefineConcept(...) decltype(___valid_expr___(__VA_ARGS__))

#define DefineTypeTraitFromConcept(TTName, ConceptName)             \
  template <typename... Args>                                       \
  struct TTName : RAJA::concepts::requires_<ConceptName, Args...> { \
  }

namespace RAJA
{

namespace concepts
{

namespace detail
{

template <class...>
struct TL {
};

template <class...>
struct voider {
  using type = void;
};

template <class Default,
          class /* always void*/,
          template <class...> class Concept,
          class TArgs>
struct detector {
  using value_t = metalib::false_type;
  using type = Default;
};

template <class Default, template <class...> class Concept, class... Args>
struct detector<Default,
                typename voider<Concept<Args...>>::type,
                Concept,
                TL<Args...>> {
  using value_t = metalib::true_type;
  using type = Concept<Args...>;
};

template <template <class...> class Concept, class TArgs>
using is_detected = detector<void, void, Concept, TArgs>;

template <template <class...> class Concept, class TArgs>
using detected = typename is_detected<Concept, TArgs>::value_t;


template <typename Ret, typename T>
Ret returns(T const &) noexcept;

}  // end namespace detail

template <typename T>
using negate = metalib::negate_t<T>;

using concepts::metalib::bool_;

/// metafunction to get instance of value type for concepts
template <typename T>
auto val() noexcept -> decltype(std::declval<T>());

/// metafunction to get instance of const type for concepts
template <typename T>
auto cval() noexcept -> decltype(std::declval<T const>());

/// metafunction for use within decltype expression to validate return type is
/// convertible to given type
template <typename T, typename U>
constexpr auto convertible_to(U &&u) noexcept
    -> decltype(detail::returns<metalib::true_type>(static_cast<T>((U &&) u)));

/// metafunction for use within decltype expression to validate type of
/// expression
template <typename T, typename U>
constexpr auto has_type(U &&) noexcept
    -> metalib::if_<metalib::is_same<T, U>, metalib::true_type>;

template <typename BoolLike>
constexpr auto is(BoolLike) noexcept
    -> metalib::if_<BoolLike, metalib::true_type>;

template <typename BoolLike>
constexpr auto is_not(BoolLike) noexcept
    -> metalib::if_c<!BoolLike::value, metalib::true_type>;

/// metaprogramming concept for SFINAE checking of aggregating concepts
template <typename... Args>
struct all_of : metalib::all_of_t<Args...> {
};

/// metaprogramming concept for SFINAE checking of aggregating concepts
template <typename... Args>
struct none_of : metalib::none_of_t<Args...> {
};

/// metaprogramming concept for SFINAE checking of aggregating concepts
template <typename... Args>
struct any_of : metalib::any_of_t<Args...> {
};

/// SFINAE multiple type traits
template <typename... Args>
using enable_if = typename std::enable_if<all_of<Args...>::value, void>::type;

/// SFINAE concept checking
template <template <class...> class Op, class... Args>
struct requires_ : detail::detected<Op, detail::TL<Args...>> {
};

namespace types
{

template <typename T>
using decay_t =
    typename std::remove_reference<typename std::remove_cv<T>::type>::type;

template <typename T>
using plain_t = typename std::remove_reference<T>::type;

template <typename T>
using diff_t = decltype(val<plain_t<T>>() - val<plain_t<T>>());

template <typename T>
using iterator_t = decltype(std::begin(val<plain_t<T>>()));

}  // end namespace types

template <typename T>
struct LessThanComparable
    : DefineConcept(convertible_to<bool>(val<T>() < val<T>())) {
};

template <typename T>
struct GreaterThanComparable
    : DefineConcept(convertible_to<bool>(val<T>() > val<T>())) {
};

template <typename T>
struct LessEqualComparable
    : DefineConcept(convertible_to<bool>(val<T>() <= val<T>())) {
};

template <typename T>
struct GreaterEqualComparable
    : DefineConcept(convertible_to<bool>(val<T>() >= val<T>())) {
};

template <typename T>
struct EqualityComparable
    : DefineConcept(convertible_to<bool>(val<T>() == val<T>())) {
};

template <typename T, typename U>
struct ComparableTo
    : DefineConcept(convertible_to<bool>(val<U>() < val<T>()),
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
                    convertible_to<bool>(val<T>() != val<U>())) {
};

template <typename T>
struct Comparable : ComparableTo<T, T> {
};

template <typename T>
struct Arithmetic : DefineConcept(is(std::is_arithmetic<T>())) {
};

template <typename T>
struct FloatingPoint : DefineConcept(is(std::is_floating_point<T>())) {
};

template <typename T>
struct Integral : DefineConcept(is(std::is_integral<T>())) {
};

template <typename T>
struct Signed : DefineConcept(Integral<T>(), is(std::is_signed<T>())) {
};

template <typename T>
struct Unsigned : DefineConcept(Integral<T>(), is(std::is_unsigned<T>())) {
};

template <typename T>
struct Iterator
    : DefineConcept(is_not(Integral<T>()),  // hacky NVCC 8 workaround
                    *(val<T>()),
                    has_type<T &>(++val<T &>())) {
};

template <typename T>
struct ForwardIterator
    : DefineConcept(Iterator<T>(), val<T &>()++, *val<T &>()++) {
};

template <typename T>
struct BidirectionalIterator
    : DefineConcept(ForwardIterator<T>(),
                    has_type<T &>(--val<T &>()),
                    convertible_to<T const &>(val<T &>()--),
                    *val<T &>()--) {
};

template <typename T>
struct RandomAccessIterator
    : DefineConcept(BidirectionalIterator<T>(),
                    Comparable<T>(),
                    has_type<T &>(val<T &>() += val<types::diff_t<T>>()),
                    has_type<T>(val<T>() + val<types::diff_t<T>>()),
                    has_type<T>(val<types::diff_t<T>>() + val<T>()),
                    has_type<T &>(val<T &>() -= val<types::diff_t<T>>()),
                    has_type<T>(val<T>() - val<types::diff_t<T>>()),
                    val<T>()[val<types::diff_t<T>>()]) {
};

template <typename T>
struct HasBeginEnd : DefineConcept(std::begin(val<T>()), std::end(val<T>())) {
};

template <typename T>
struct Range
    : DefineConcept(HasBeginEnd<T>(), Iterator<types::iterator_t<T>>()) {
};

template <typename T>
struct ForwardRange
    : DefineConcept(HasBeginEnd<T>(), ForwardIterator<types::iterator_t<T>>()) {
};

template <typename T>
struct BidirectionalRange
    : DefineConcept(HasBeginEnd<T>(),
                    BidirectionalIterator<types::iterator_t<T>>()) {
};

template <typename T>
struct RandomAccessRange
    : DefineConcept(HasBeginEnd<T>(),
                    RandomAccessIterator<types::iterator_t<T>>()) {
};

}  // end namespace concepts

namespace type_traits
{
DefineTypeTraitFromConcept(is_iterator, RAJA::concepts::Iterator);
DefineTypeTraitFromConcept(is_forward_iterator,
                           RAJA::concepts::ForwardIterator);
DefineTypeTraitFromConcept(is_bidirectional_iterator,
                           RAJA::concepts::BidirectionalIterator);
DefineTypeTraitFromConcept(is_random_access_iterator,
                           RAJA::concepts::RandomAccessIterator);

DefineTypeTraitFromConcept(is_range, RAJA::concepts::Range);
DefineTypeTraitFromConcept(is_forward_range, RAJA::concepts::ForwardRange);
DefineTypeTraitFromConcept(is_bidirectional_range,
                           RAJA::concepts::BidirectionalRange);
DefineTypeTraitFromConcept(is_random_access_range,
                           RAJA::concepts::RandomAccessRange);

DefineTypeTraitFromConcept(is_comparable, RAJA::concepts::Comparable);
DefineTypeTraitFromConcept(is_comparable_to, RAJA::concepts::ComparableTo);

DefineTypeTraitFromConcept(is_arithmetic, RAJA::concepts::Arithmetic);
DefineTypeTraitFromConcept(is_floating_point, RAJA::concepts::FloatingPoint);
DefineTypeTraitFromConcept(is_integral, RAJA::concepts::Integral);
DefineTypeTraitFromConcept(is_signed, RAJA::concepts::Signed);
DefineTypeTraitFromConcept(is_unsigned, RAJA::concepts::Unsigned);

template <typename T>
using IterableValue = decltype(*std::begin(RAJA::concepts::val<T>()));

template <typename T>
using IteratorValue = decltype(*RAJA::concepts::val<T>());

namespace detail
{

template <typename, template <typename...> class, typename...>
struct IsSpecialized : RAJA::concepts::metalib::false_type {
};

template <template <typename...> class Template, typename... T>
struct IsSpecialized<typename concepts::detail::voider<decltype(
                         concepts::val<Template<T...>>())>::type,
                     Template,
                     T...> : RAJA::concepts::metalib::true_type {
};

template <template <class...> class, template <class...> class, bool, class...>
struct SpecializationOf : RAJA::concepts::metalib::false_type {
};

template <template <class...> class Expected,
          template <class...> class Actual,
          class... Args>
struct SpecializationOf<Expected, Actual, true, Args...>
    : RAJA::concepts::metalib::is_same<Expected<Args...>, Actual<Args...>> {
};

}  // end namespace detail


template <template <class...> class Outer, class... Args>
using IsSpecialized = detail::IsSpecialized<void, Outer, Args...>;

template <template <class...> class, typename T>
struct SpecializationOf : RAJA::concepts::metalib::false_type {
};

template <template <class...> class Expected,
          template <class...> class Actual,
          class... Args>
struct SpecializationOf<Expected, Actual<Args...>>
    : detail::SpecializationOf<Expected,
                               Actual,
                               IsSpecialized<Expected, Args...>::value,
                               Args...> {
};

}  // end namespace type_traits

}  // end namespace RAJA

#endif
