/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file with support for pre-C++14 compilers.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_LEGACY_COMPATIBILITY_HPP
#define RAJA_LEGACY_COMPATIBILITY_HPP

#include "RAJA/config.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <type_traits>
#include <utility>

#include "camp/camp.hpp"

#include "RAJA/util/macros.hpp"

#if (!defined(CAMP_HAS_FOLD_EXPRESSIONS)) && \
    defined(__cpp_fold_expressions) && __cpp_fold_expressions >= 201603
#define CAMP_HAS_FOLD_EXPRESSIONS 1
#endif


#if (!defined(__INTEL_COMPILER)) && (!defined(RAJA_COMPILER_MSVC))
static_assert(__cplusplus >= 201103L,
              "C++ standards below 2011 are not "
              "supported" RAJA_STRINGIFY_HELPER(__cplusplus));
#endif

#include <array>
#include <tuple>
namespace VarOps
{
using std::get;
using std::make_tuple;
using std::tuple;
using std::tuple_cat;
using std::tuple_element;
using std::tuple_size;
}  // namespace VarOps
// #endif

namespace VarOps
{

// Basics, using c++14 semantics in a c++11 compatible way, credit to libc++

// Forward

// FoldL

template <typename Op, typename Arg1>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto foldl(
    Op&& RAJA_UNUSED_ARG(operation),
    Arg1&& arg)
{
  return camp::forward<Arg1>(arg);
}

template <typename Op, typename Arg1, typename Arg2>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto foldl(Op&& operation,
                                                  Arg1&& arg1,
                                                  Arg2&& arg2)
{
  return camp::forward<Op>(operation)(camp::forward<Arg1>(arg1),
                                      camp::forward<Arg2>(arg2));
}

template <typename Op,
          typename Arg1,
          typename Arg2,
          typename Arg3,
          typename... Rest>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto foldl(Op&& operation,
                                                  Arg1&& arg1,
                                                  Arg2&& arg2,
                                                  Arg3&& arg3,
                                                  Rest&&... rest)
{
  return foldl(camp::forward<Op>(operation),
               camp::forward<Op>(operation)(
                   camp::forward<Op>(operation)(camp::forward<Arg1>(arg1),
                                                camp::forward<Arg2>(arg2)),
                   camp::forward<Arg3>(arg3)),
               camp::forward<Rest>(rest)...);
}


// Convenience folds
template <typename Result, typename... Args>
RAJA_HOST_DEVICE RAJA_INLINE constexpr Result sum(Args... args)
{
#ifdef CAMP_HAS_FOLD_EXPRESSIONS
  return (... + args);
#else
  return foldl(RAJA::operators::plus<Result>(), args...);
#endif
}

template <typename Result, typename... Args>
RAJA_HOST_DEVICE RAJA_INLINE constexpr Result max(Args... args)
{
  return std::max({args...});
}

template <typename Result, typename... Args>
RAJA_HOST_DEVICE RAJA_INLINE constexpr Result min(Args... args)
{
  return std::min({args...});
}

template <typename Result, typename... Args>
RAJA_HOST_DEVICE RAJA_INLINE constexpr Result product(Args... args)
{
#ifdef CAMP_HAS_FOLD_EXPRESSIONS
  return (... * args);
#else
  return foldl(RAJA::operators::multiplies<Result>(), args...);
#endif
}

template <template <class...> class Seq, class First, class... Ints>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto rotate_left_one(
    const Seq<First, Ints...>)
{
  return Seq<Ints..., First>{};
}


// Index sequence
template <size_t Upper>
using make_index_sequence = typename camp::make_int_seq<size_t, Upper>::type;

template <size_t... Ints>
using index_sequence = camp::int_seq<size_t, Ints...>;

// Invoke
using camp::invoke;
using camp::invoke_with_order;

// Ignore helper
template <typename... Args>
RAJA_HOST_DEVICE RAJA_INLINE void ignore_args(Args...)
{
}

// Assign

template <size_t... To, size_t... From, typename ToT, typename FromT>
RAJA_HOST_DEVICE RAJA_INLINE void assign(ToT&& dst,
                                         FromT src,
                                         index_sequence<To...>,
                                         index_sequence<From...>)
{
  ignore_args((dst[To] = src[From])...);
}

template <size_t... To, typename ToT, typename... Args>
RAJA_HOST_DEVICE RAJA_INLINE void assign_args(ToT&& dst,
                                              index_sequence<To...>,
                                              Args... args)
{
  ignore_args((dst[To] = args)...);
}

}  // namespace VarOps

#endif
