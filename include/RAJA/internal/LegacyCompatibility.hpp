/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file with support for pre-C++14 compilers.
 *
 ******************************************************************************
 */

#ifndef RAJA_LEGACY_COMPATIBILITY_HPP
#define RAJA_LEGACY_COMPATIBILITY_HPP

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

#include "RAJA/config.hpp"

#include "RAJA/util/defines.hpp"

#if (!defined(__INTEL_COMPILER)) && (!defined(RAJA_COMPILER_MSVC))
static_assert(__cplusplus >= 201103L,
              "C++ standards below 2011 are not "
              "supported" RAJA_STRINGIFY_HELPER(__cplusplus));
#endif

#include <cstdint>
#include <functional>
#include <iostream>
#include <type_traits>
#include <utility>

#if __cplusplus > 201400L
#define RAJA_CXX14_CONSTEXPR constexpr
#else
#define RAJA_CXX14_CONSTEXPR
#endif

// #if defined(RAJA_USE_CUDA)
// #include <thrust/tuple.h>
// namespace VarOps {
//     using thrust::tuple;
//     using thrust::tuple_element;
//     using thrust::get;
//     using thrust::tuple_size;
//     using thrust::make_tuple;
// }
// #else
#include <array>
#include <tuple>
namespace VarOps
{
using std::tuple;
using std::tuple_element;
using std::tuple_cat;
using std::get;
using std::tuple_size;
using std::make_tuple;
}
// #endif

namespace VarOps
{

// Basics, using c++14 semantics in a c++11 compatible way, credit to libc++

// Forward
template <class T>
struct remove_reference {
  typedef T type;
};
template <class T>
struct remove_reference<T&> {
  typedef T type;
};
template <class T>
struct remove_reference<T&&> {
  typedef T type;
};
template <class T>
RAJA_HOST_DEVICE RAJA_INLINE constexpr T&& forward(
    typename remove_reference<T>::type& t) noexcept
{
  return static_cast<T&&>(t);
}
template <class T>
RAJA_HOST_DEVICE RAJA_INLINE constexpr T&& forward(
    typename remove_reference<T>::type&& t) noexcept
{
  return static_cast<T&&>(t);
}

// FoldL
template <typename Op, typename... Rest>
struct foldl_impl;

template <typename Op, typename Arg1>
struct foldl_impl<Op, Arg1> {
  using Ret = Arg1;
};

template <typename Op, typename Arg1, typename Arg2>
struct foldl_impl<Op, Arg1, Arg2> {
  using Ret = typename std::result_of<Op(Arg1, Arg2)>::type;
};

template <typename Op,
          typename Arg1,
          typename Arg2,
          typename Arg3,
          typename... Rest>
struct foldl_impl<Op, Arg1, Arg2, Arg3, Rest...> {
  using Ret =
      typename foldl_impl<Op,
                          typename std::result_of<Op(
                              typename std::result_of<Op(Arg1, Arg2)>::type,
                              Arg3)>::type,
                          Rest...>::Ret;
};

template <typename Op, typename Arg1>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto foldl(
    Op&& RAJA_UNUSED_ARG(operation),
    Arg1&& arg) -> typename foldl_impl<Op, Arg1>::Ret
{
  return forward<Arg1&&>(arg);
}

template <typename Op, typename Arg1, typename Arg2>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto foldl(Op&& operation,
                                                  Arg1&& arg1,
                                                  Arg2&& arg2) ->
    typename foldl_impl<Op, Arg1, Arg2>::Ret
{
  return forward<Op&&>(operation)(forward<Arg1&&>(arg1), forward<Arg2&&>(arg2));
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
                                                  Rest&&... rest) ->
    typename foldl_impl<Op, Arg1, Arg2, Arg3, Rest...>::Ret
{
  return foldl(forward<Op&&>(operation),
               forward<Op&&>(
                   operation)(forward<Op&&>(operation)(forward<Arg1&&>(arg1),
                                                       forward<Arg2&&>(arg2)),
                              forward<Arg3&&>(arg3)),
               forward<Rest&&>(rest)...);
}

struct adder {
  template <typename Result>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr Result operator()(
      const Result& l,
      const Result& r) const
  {
    return l + r;
  }
};

// Convenience folds
template <typename Result, typename... Args>
RAJA_HOST_DEVICE RAJA_INLINE constexpr Result sum(Args... args)
{
  return foldl(adder(), args...);
}

// template<typename Result, size_t N>
// struct product_first_n;
//
// template<typename Result>
// struct product_first_n<Result, 0>{
//     static Result value = 1;
//     template<typename ... Args>
//     constexpr product_first_n(Args...args) : value{1} { }
// };
//
// template<typename Result, size_t N>
// struct product_first_n{
//     static Result value = product_first_n<Result, N-1>(args...)::value;
//     template<typename FirstArg, typename ... Args>
//     constexpr product_first_n(FirstArg arg1, Args...args)
//     : value() { }
// };

// Index sequence

template <size_t... Ints>
struct integer_sequence {
  using type = integer_sequence;
  static constexpr size_t size = sizeof...(Ints);
  static constexpr std::array<size_t, sizeof...(Ints)> value{Ints...};
};

template <template <class...> class Seq, class First, class... Ints>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto rotate_left_one(
    const Seq<First, Ints...>) -> Seq<Ints..., First>
{
  return Seq<Ints..., First>{};
}

template <size_t... Ints>
constexpr size_t integer_sequence<Ints...>::size;
template <size_t... Ints>
constexpr std::array<size_t, sizeof...(Ints)> integer_sequence<Ints...>::value;

namespace integer_sequence_detail
{
// using aliases for cleaner syntax
template <class T>
using Invoke = typename T::type;

template <class S1, class S2>
struct concat;

template <size_t... I1, size_t... I2>
struct concat<integer_sequence<I1...>, integer_sequence<I2...>>
    : integer_sequence<I1..., (sizeof...(I1) + I2)...> {
};

template <class S1, class S2>
using Concat = Invoke<concat<S1, S2>>;

template <size_t N>
struct gen_seq;
template <size_t N>
using GenSeq = Invoke<gen_seq<N>>;

template <size_t N>
struct gen_seq : Concat<GenSeq<N / 2>, GenSeq<N - N / 2>> {
};

template <>
struct gen_seq<0> : integer_sequence<> {
};
template <>
struct gen_seq<1> : integer_sequence<0> {
};
}

template <size_t Upper>
using make_index_sequence =
    typename integer_sequence_detail::gen_seq<Upper>::type;

template <size_t... Ints>
using index_sequence = integer_sequence<Ints...>;

// Invoke

template <typename Fn, size_t... Sequence, typename TupleLike>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto invoke_with_order(
    TupleLike&& t,
    Fn&& f,
    index_sequence<Sequence...>) -> decltype(f(get<Sequence>(t)...))
{
  return f(get<Sequence>(t)...);
}

template <typename Fn, typename TupleLike>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto invoke(TupleLike&& t, Fn&& f)
    -> decltype(
        invoke_with_order(t,
                          f,
                          make_index_sequence<tuple_size<TupleLike>::value>{}))
{
  return invoke_with_order(t,
                           f,
                           make_index_sequence<tuple_size<TupleLike>::value>{});
}

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

// Get nth element of parameter pack
template <size_t index, size_t first, size_t... rest>
struct get_at {
  static constexpr size_t value = get_at<index - 1, rest...>::value;
};

template <size_t first, size_t... rest>
struct get_at<0, first, rest...> {
  static constexpr size_t value = first;
};

// Get nth element of parameter pack
template <size_t index, typename first, typename... rest>
struct get_type_at {
  using type = typename get_type_at<index - 1, rest...>::type;
};

template <typename first, typename... rest>
struct get_type_at<0, first, rest...> {
  using type = first;
};

// Get offset of element of parameter pack
template <size_t diff, size_t off, size_t match, size_t... rest>
struct get_offset_impl {
  static constexpr size_t value =
      get_offset_impl<match - get_at<off + 1, rest...>::value,
                      off + 1,
                      match,
                      rest...>::value;
};

template <size_t off, size_t match, size_t... rest>
struct get_offset_impl<0, off, match, rest...> {
  static constexpr size_t value = off;
};

template <size_t match, size_t first, size_t... rest>
struct get_offset
    : public get_offset_impl<match - first, 0, match, first, rest...> {
};

// Get nth element of argument list
// TODO: add specializations to make this compile faster and with less
// recursion
template <size_t index>
struct get_arg_at {
  template <typename First, typename... Rest>
  RAJA_HOST_DEVICE RAJA_INLINE static constexpr auto value(
      First&& RAJA_UNUSED_ARG(first),
      Rest&&... rest)
      -> decltype(VarOps::forward<
                  typename VarOps::get_type_at<index - 1, Rest...>::type>(
          get_arg_at<index - 1>::value(VarOps::forward<Rest>(rest)...)))
  {
    static_assert(index < sizeof...(Rest) + 1, "index is past the end");
    return VarOps::forward<
        typename VarOps::get_type_at<index - 1, Rest...>::type>(
        get_arg_at<index - 1>::value(VarOps::forward<Rest>(rest)...));
  }
};

template <>
struct get_arg_at<0> {
  template <typename First, typename... Rest>
  RAJA_HOST_DEVICE RAJA_INLINE static constexpr auto value(
      First&& first,
      Rest&&... RAJA_UNUSED_ARG(rest))
      -> decltype(VarOps::forward<First>(first))
  {
    return VarOps::forward<First>(first);
  }
};
}

#endif
