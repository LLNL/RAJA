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

#include "RAJA/internal/detector.hpp"
#include "RAJA/internal/metalib.hpp"

namespace RAJA
{

namespace concepts
{

/// metafunction to get instance of value type for concepts
template <typename T>
auto val() -> decltype(std::declval<T>())
{
  return std::declval<T>();
}

/// metafunction to get instance of const type for concepts
template <typename T>
auto cval() -> decltype(std::declval<T const>())
{
  return std::declval<T const>();
}

/// metafunction to get instance of reference type for concepts
template <typename T>
auto ref() -> decltype(std::declval<T &>())
{
  return std::declval<T &>();
}

/// metafunction to get instance of const reference type for concepts
template <typename T>
auto cref() -> decltype(std::declval<T const &>())
{
  return std::declval<T const &>();
}

using metalib::convertible_to;
using metalib::has_type;
using metalib::models;
using metalib::valid_expr;

template <bool Bool>
using bool_ = std::integral_constant<bool, Bool>;

template <typename... BoolLike>
auto conforms()
    -> metalib::if_<metalib::all_of_t<BoolLike...>, std::true_type>;

/// metaprogramming concept for SFINAE checking of concepts
template <template <typename...> class Thing, typename... Args>
using requires_ = is_detected<Thing, Args...>;

/// metaprogramming concept for SFINAE checking of aggregating concepts
template <typename... Args>
using all_of = metalib::all_of_t<Args...>;

template <typename... Args>
using none_of = metalib::none_of_t<Args...>;

template <typename... Args>
using any_of = metalib::any_of_t<Args...>;

template <typename... Args>
using enable_if = typename std::enable_if<all_of<Args...>::value>::type;

}  // end namespace concepts

}  // end namespace RAJA

/// Convenience macro which wraps `decltype(concepts::valid_expr( ... ))`
#define DefineConcept(...) decltype(RAJA::concepts::valid_expr(__VA_ARGS__))

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
using diff_t = decltype(val<T>() - val<T>());
template <typename T>
using iterator_t = decltype(ref<T>().begin());

template <typename T>
using Iterator = DefineConcept(*val<T>(), has_type<T &>(++ref<T>()));

template <typename T>
using ForwardIterator = DefineConcept(models<Iterator<T>>(),
                                      ref<T>()++,
                                      *ref<T>()++);

template <typename T>
using BidirectionalIterator =
    DefineConcept(models<ForwardIterator<T>>(),
                  has_type<T &>(--ref<T>()),
                  convertible_to<T const &>(ref<T>()--),
                  *ref<T>()--);

template <typename T>
using RandomAccessIterator =
    DefineConcept(models<BidirectionalIterator<T>>(),
                  models<Comparable<T>>(),
                  has_type<T &>(ref<T>() += val<diff_t<T>>()),
                  has_type<T>(val<T>() + val<diff_t<T>>()),
                  has_type<T>(val<diff_t<T>>() + val<T>()),
                  has_type<T &>(ref<T>() -= val<diff_t<T>>()),
                  has_type<T>(val<T>() - val<diff_t<T>>()),
                  val<T>()[val<diff_t<T>>()]);

template <typename T>
using HasMemberBegin = DefineConcept(ref<T>().begin());

template <typename T>
using HasMemberEnd = DefineConcept(ref<T>().end());

template <typename T>
using HasBeginEnd = DefineConcept(models<HasMemberBegin<T>>(),
                                  models<HasMemberEnd<T>>());

template <typename T>
using Range = DefineConcept(models<HasBeginEnd<T>>(),
                            models<Iterator<iterator_t<T>>>());

template <typename T>
using ForwardRange = DefineConcept(models<HasBeginEnd<T>>(),
                                   models<ForwardIterator<iterator_t<T>>>());

template <typename T>
using BidirectionalRange =
    DefineConcept(models<HasBeginEnd<T>>(),
                  models<BidirectionalIterator<iterator_t<T>>>());

template <typename T>
using RandomAccessRange =
    DefineConcept(models<HasBeginEnd<T>>(),
                  models<RandomAccessIterator<iterator_t<T>>>());

template <typename T>
using Integral = DefineConcept(conforms<std::is_integral<T>>());

template <typename T>
using Signed = DefineConcept(models<Integral<T>>(),
                             conforms<std::is_signed<T>>());
template <typename T>
using Unsigned = DefineConcept(models<Integral<T>>(),
                               conforms<std::is_unsigned<T>>());

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
